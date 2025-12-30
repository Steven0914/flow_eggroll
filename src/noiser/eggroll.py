import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Dict, Tuple, List, Optional, Union

# ==========================================
# 1. Global Noise Table (For Fast RNG)
# ==========================================
class NoiseTable:
    """
    Device-aware Singleton pattern to support multi-GPU environments.
    Efficiently slices noise from a large pre-allocated table.
    """
    _instances: Dict[torch.device, 'NoiseTable'] = {}
    
    def __new__(cls, size=10_000_000, device='cuda', dtype=torch.float32):
        if isinstance(device, str):
            device = torch.device(device)
            
        if device not in cls._instances:
            instance = super(NoiseTable, cls).__new__(cls)
            # print(f"Initializing Global Noise Table on {device} ({size} params)...")
            # We use a large prime seed for the table itself to ensure randomness
            gen = torch.Generator(device=device)
            gen.manual_seed(1337) 
            instance.table = torch.randn(size, device=device, dtype=dtype, generator=gen)
            instance.size = size
            instance.device = device
            cls._instances[device] = instance
            
        return cls._instances[device]

    @classmethod
    def get_slice(cls, seed: int, numel: int, device: torch.device):
        """
        Returns a 1D slice of noise of length `numel` deterministically based on `seed`.
        """
        instance = cls(device=device) 
        # Simple deterministic hashing for start index. 
        # Must be robust enough to avoid overlapping slices for different seeds too frequently.
        # Current constants are arbitrary primes.
        start_idx = (seed * 123456789 + 987654321) % (instance.size - numel)
        return instance.table[start_idx : start_idx + numel]

# ==========================================
# 2. EggrollLinear (Custom Layer)
# ==========================================
class EggrollLinear(nn.Module):
    """
    A Linear layer that supports implicit low-rank perturbation.
    It wraps an existing nn.Linear layer.
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 1, config: Dict = None):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # We share the weight/bias with the original layer (sanity check: make sure they are preserved)
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        
        self.rank = rank
        self.config = config if config is not None else {}
        # Default config
        self.config.setdefault("sigma", 1e-3)
        self.config.setdefault("noise_reuse", 0)
        
        self.current_iterinfo: Optional[torch.Tensor] = None # Tensor([epoch, sample_idx])
        self.layer_id = 0 # Will be set by EggRoll manager

    def set_iterinfo(self, iterinfo: torch.Tensor):
        """Called by EggRoll manager before forward pass."""
        self.current_iterinfo = iterinfo

    def clear_iterinfo(self):
        """Called by EggRoll manager after forward pass."""
        self.current_iterinfo = None

    def _get_ab(self, epoch: int, thread_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstructs A and B matrices for a specific epoch and thread_id (sample_idx).
        """
        # Noise Reuse logic
        noise_reuse = self.config["noise_reuse"]
        true_epoch = 0 if noise_reuse == 0 else epoch // noise_reuse
        
        # Unique seed generation
        # logic must match JAX: fold_in(key, epoch), fold_in(..., thread_id)
        # We approximate this with linear combination of primes
        true_thread_idx = thread_id // 2
        
        # To match the "sigma sign flipping" Antithetic Sampling of standard ES if desired.
        # However, EggRoll paper (Implicit Low Rank) typically uses random A, B.
        # But let's follow the standard antithetic pattern if that's what JAX code does.
        # JAX code: sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)
        sigma_base = self.config["sigma"]
        sigma = sigma_base if thread_id % 2 == 0 else -sigma_base
        
        # Scale sigma by sqrt(rank) as per JAX: base_sigma / jnp.sqrt(rank)
        # Reason: A @ B.T sums 'rank' random variables. To keep variance consistent, we scale.
        scaled_sigma = sigma / math.sqrt(self.rank)

        # Unique seed for this layer and this step
        # layer_id ensures different layers get different noise
        unique_seed = true_epoch * 1_000_000 + true_thread_idx + (self.layer_id * 7919)
        
        numel_a = self.out_features * self.rank
        numel_b = self.in_features * self.rank
        total_numel = numel_a + numel_b
        
        raw_noise = NoiseTable.get_slice(unique_seed, total_numel, self.weight.device)
        
        # Slicing A and B
        # JAX: lora_params = ...; B = [:b], A = [b:]
        # Here we follow same order
        b_slice = raw_noise[:numel_b].view(self.in_features, self.rank)
        a_slice = raw_noise[numel_b:].view(self.out_features, self.rank)
        
        # Apply sigma scaling to A (arbitrary choice, conceptually A*sigma * B)
        # Cast to correct dtype to match input (e.g. bfloat16)
        a_final = (a_slice * scaled_sigma).to(self.weight.dtype)
        b_final = b_slice.to(self.weight.dtype)
        
        return a_final, b_final

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Standard Linear Pass
        base_out = F.linear(input, self.weight, self.bias)
        
        # If no perturbation, return base output
        if self.current_iterinfo is None:
            return base_out

        # Local Low-Rank Perturbation
        # x(W + AB^T)^T = xW^T + x(AB^T)^T = xW^T + xBA^T = xW^T + (xB)A^T
        
        epoch = self.current_iterinfo[0].item()
        thread_id = self.current_iterinfo[1].item()
        
        A, B = self._get_ab(epoch, thread_id)
        
        # Optimized Forward: (x @ B) @ A.T
        # input: (..., in_features)
        # B: (in_features, rank)
        # A: (out_features, rank)
        
        # 1. x @ B -> (..., rank)
        x_b = input @ B
        
        # 2. (x @ B) @ A.T -> (..., out_features)
        lora_out = x_b @ A.T
        
        return base_out + lora_out

# ==========================================
# 3. EggRoll Manager
# ==========================================
class EggRoll:
    def __init__(self, model: nn.Module, target_modules: List[str], sigma: float, lr: float, group_size=0, noise_reuse=0, rank=1, optimizer_cls=optim.SGD, **optimizer_kwargs):
        """
        Args:
            model: The base model (e.g. pipeline.transformer)
            target_modules: List of module names to replace (e.g. ["attn.to_q", ...])
            sigma: Noise strength
            lr: Learning rate
            rank: Rank for implicit perturbation
        """
        self.model = model
        self.sigma = sigma
        self.lr = lr
        self.rank = rank
        self.config = {
            "sigma": sigma,
            "noise_reuse": noise_reuse,
            "group_size": group_size,
        }
        
        self.replaced_modules: List[EggrollLinear] = []
        self._replace_modules(model, target_modules)
        
        # Optimize only the weights of the replaced modules (which are the original weights)
        # Note: In standard ES, we optimize ALL perturbed parameters.
        # Since we modify the original weights in-place during update, we pass them to optimizer.
        params_to_optimize = [m.weight for m in self.replaced_modules]
        if len(params_to_optimize) == 0:
            print("Warning: No modules replaced by EggRoll!")
            
        self.device = params_to_optimize[0].device if params_to_optimize else torch.device('cuda')
        NoiseTable(device=self.device) # Init table
        
        self.optimizer = optimizer_cls(params_to_optimize, lr=lr, **optimizer_kwargs)

    def _replace_modules(self, model, target_substrings):
        """
        Traverses model and replaces nn.Linear layers whose names match target_substrings.
        Similar to PEFT's mechanisms but simpler.
        """
        print(f"EggRoll: Replacing layers checking for substrings: {target_substrings}")
        count = 0
        for name, module in model.named_modules():
            # Check if module name ends with any of the target substrings (standard LoRA behavior)
            # or if any target is in name. Let's be reasonably specific.
            is_target = False
            for target in target_substrings:
                if name.endswith(target):
                    is_target = True
                    break
            
            if is_target and isinstance(module, nn.Linear):
                # Replace
                self._replace_module_in_parent(model, name, module)
                count += 1

        print(f"EggRoll: Replaced {count} layers with EggrollLinear (Rank={self.rank}).")
        
        # Assign layer IDs for unique noise generation
        for i, m in enumerate(self.replaced_modules):
            m.layer_id = i

    def _replace_module_in_parent(self, root_model, module_name, old_module):
        # We need to find the parent to setattr
        # name is like "transformer.blocks.0.attn.to_q"
        atoms = module_name.split(".")
        parent = root_model
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        
        target_name = atoms[-1]
        
        # Create new EggrollLinear
        new_module = EggrollLinear(old_module, rank=self.rank, config=self.config)
        
        # Replace
        setattr(parent, target_name, new_module)
        self.replaced_modules.append(new_module)

    def perturb(self, iterinfo: torch.Tensor):
        """
        Instead of modifying weights, we just set the state of EggrollLinears.
        They will add noise during forward pass.
        """
        for m in self.replaced_modules:
            m.set_iterinfo(iterinfo)

    def unperturb(self, iterinfo: torch.Tensor):
        """
        Clear state.
        """
        for m in self.replaced_modules:
            m.clear_iterinfo()

    @staticmethod
    def convert_fitnesses(raw_scores: torch.Tensor, group_size: int = 0):
        # Same Rank Normalization Logic
        if group_size == 0:
            mean = torch.mean(raw_scores, dim=0, keepdim=True)
            std = torch.std(raw_scores, dim=0, keepdim=True) + 1e-5
            true_scores = (raw_scores - mean) / std
        else:
            group_scores = raw_scores.view(-1, group_size)
            mean = torch.mean(group_scores, dim=-1, keepdim=True)
            std = torch.std(group_scores, dim=-1, keepdim=True) + 1e-5
            true_scores = (group_scores - mean) / std
            true_scores = true_scores.view(-1)
        return true_scores

    def step(self, fitnesses: torch.Tensor, iterinfos: torch.Tensor):
        """
        Performs the Update step.
        Reconstructs noise and calculates gradients.
        """
        scores = self.convert_fitnesses(fitnesses, self.config["group_size"])
        self.optimizer.zero_grad()
        
        pop_size = scores.shape[0]
        chunk_size = 1000 # Process in chunks to save memory
        
        # We iterate over each module (parameter tensor)
        # This is where we reconstruct A and B
        
        for m in self.replaced_modules:
            param = m.weight
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            
            # Reconstruction Loop
            # We want to compute: Grad = - (1/sqrt(N)) * Sum_i ( Score_i * E_i )
            # Where E_i = A_i @ B_i^T
            # So Grad = - (1/sqrt(N)) * Sum_i ( Score_i * A_i @ B_i^T )
            
            # Using chunks for vectorization
            for start in range(0, pop_size, chunk_size):
                end = min(start + chunk_size, pop_size)
                chunk_inds = iterinfos[start:end] # (Chunk, 2)
                chunk_scores = scores[start:end]   # (Chunk,)
                
                # Batch generate A and B
                # We need a vectorized version of m._get_ab or just loop for now if too complex to vectorize cleanly
                # Given strict requirements, let's loop inside chunk or optimize get_slice
                # Since get_slice creates a new tensor, simple python loop collecting list is fine for batch size ~1000
                
                a_list = []
                b_list = []
                
                for k in range(end - start):
                    epoch = chunk_inds[k][0].item()
                    th_id = chunk_inds[k][1].item()
                    a, b = m._get_ab(epoch, th_id)
                    # Cast to float32 for gradient computation precision and to avoid mismatch with scores (float32)
                    a_list.append(a.float())
                    b_list.append(b.float())
                
                stack_a = torch.stack(a_list) # (Chunk, out, r)
                stack_b = torch.stack(b_list) # (Chunk, in, r)
                
                # Weight A by scores
                # stack_a: (Chunk, Out, Rank)
                # scores: (Chunk,) -> (Chunk, 1, 1)
                weighted_a = stack_a * chunk_scores.view(-1, 1, 1)
                
                # Compute Sum_i ( (Score_i * A_i) @ B_i^T )
                # using einsum: "nor,nir->oi" where n=batch, o=out, i=in, r=rank
                # weighted_a (n, o, r)
                # stack_b (n, i, r)
                # Output (o, i) which matches weight shape (out, in)
                
                chunk_grad_sum = torch.einsum('nor,nir->oi', weighted_a, stack_b)
                
                # Accumulate negative gradient (since we want maximizing reward)
                # chunk_grad_sum is likely float32 (due to earlier casting)
                # param.grad is likely bfloat16 (model weight dtype)
                # We explicitly cast to param.grad's dtype to ensure safe in-place addition.
                if param.grad.dtype != chunk_grad_sum.dtype:
                     param.grad.add_(-chunk_grad_sum.to(dtype=param.grad.dtype))
                else:
                     param.grad.add_(-chunk_grad_sum)

            # Final scaling by 1/sqrt(N)
            param.grad.div_(math.sqrt(pop_size))
            
        self.optimizer.step()