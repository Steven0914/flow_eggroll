import os
import sys
import datetime
from typing import List
from absl import app, flags
from ml_collections import config_flags
from typing import List
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from diffusers import StableDiffusion3Pipeline

import wandb
import random
import numpy as np
import tempfile
from PIL import Image
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow_grpo"))
import flow_grpo.rewards as flow_rewards
from flow_grpo.ema import EMAModuleWrapper

from noiser.eggroll import EggRoll

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextPromptDataset(Dataset):
    def __init__(self, dataset_root: str, split: str = "train") -> None:
        self.file_path = os.path.join(dataset_root, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [e["prompt"] for e in examples]
        metadatas = [e["metadata"] for e in examples]
        return prompts, metadatas

def compute_softmax_ranks(fitnesses, temperature=0.1):
    """
    Transforms fitnesses into centered ranks, then applies softmax weighting.
    temperature: Controls the "sharpness" of the weighting.
                 - High T (>1.0): Uniform-like (Like simple averaging)
                 - Low T (<0.2): Greedy (Focus heavily on top performers)
    """
    x = fitnesses.clone().detach()
    pop_size = x.numel()

    # 1. Calculate Ranks (0 to N-1)
    # double argsort gives us the rank of each element
    ranks = torch.argsort(torch.argsort(x)).float()

    # 2. Normalize Ranks to [-0.5, 0.5]
    centered_ranks = ranks / (pop_size - 1) - 0.5
    
    # 3. Softmax Weighting (Sharpening)
    # We apply softmax to (rank / temperature).
    # Since centered_ranks are in [-0.5, 0.5], strict softmax might be too skewed if T is very small.
    # But that's the point.
    
    # Note: Softmax output sums to 1. 
    # To keep gradient scale consistent with standard ES (which sums roughly to 1 effectively via averaging),
    # we might need to scale it up by pop_size.
    # Standard ES Update: 1/N * sum(noise * rank)
    # Softmax Update: sum(noise * weight) where weight = exp/sum_exp
    # To match magnitude: weight should be approx 1/N. so N * weight gives us the "relative leverage".
    
    weights = torch.softmax(centered_ranks / temperature, dim=0)
    
    # 4. Center the weights (Zero-Mean Assumption for Antithetic)
    # For antithetic noise (+e, -e), we implicitly assume weights sum to something?
    # Actually, for standard params update: theta_new = theta + lr * 1/(sigma*N) * sum(reward * noise)
    # If reward is constant, sum(noise) -> 0. So centering is good but not strictly required if large N.
    # However, centering ensures that if all rewards are equal, update is zero.
    
    weights = weights - weights.mean()
    
    # 5. Rescale to recover magnitude
    # We want "average" weight to be roughly comparable to the rank scale [-0.5, 0.5]
    # Current weights are approx [-1/N, 1/N] (centered).
    # Let's scale by pop_size to make them order of [-1, 1].
    weights = weights * pop_size
    
    # Finally scale it down a bit to match the previous [-0.5, 0.5] range roughly?
    # Previous: -0.5 to 0.5. Range size = 1.
    # New: -1 to 1 (approx). Range size = 2.
    # Let's divide by 2 to keep LR tuning consistent.
    weights = weights / 2.0
    
    return weights

def train_sd3_eggroll(config) -> None:
    accelerator = Accelerator()
    device = accelerator.device

    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    
    # Update config.run_name to include timestamp
    if config.run_name:
        config.run_name = f"{config.run_name}_{timestamp}"
    else:
        config.run_name = timestamp

    if config.use_wandb and accelerator.is_main_process:
        wandb.init(
            project="flow_eggroll",
            name=config.run_name,
            config=config.to_dict(),
        )
    accelerator.wait_for_everyone()

    # Print Key Configs
    if accelerator.is_main_process:
        print("="*40)
        print("       EggRoll Concept Training")
        print("="*40)
        print(f"EggRoll Sigma: {getattr(config, 'eggroll_sigma', 1e-3)}")
        print(f"Learning Rate: {getattr(config.train, 'learning_rate', 1e-4)}")
        print(f"EggRoll Rank : {getattr(config, 'eggroll_rank', 4)}")
        print(f"Reward Fn    : {config.reward_fn}")
        
        # Population Calculation Log
        batches_per_epoch = config.sample.num_batches_per_epoch
        per_gpu_pop = batches_per_epoch * config.sample.train_batch_size
        num_gpus = accelerator.num_processes
        total_pop = per_gpu_pop * num_gpus
        
        print("-" * 40)
        print(f"Total Population : {total_pop} ({num_gpus} GPUs x {per_gpu_pop} per GPU)")
        print(f"Per-GPU Details  : {per_gpu_pop} samples (Batches: {batches_per_epoch} x BS: {config.sample.train_batch_size})")
        print("="*40)
    
    # Reproducibility
    set_seed(getattr(config, "seed", 42), device_specific=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,
        torch_dtype=torch.bfloat16
    )
    # Freeze base model
    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)

    pipeline.to(device)
    pipeline.safety_checker = None
    
    pipeline.set_progress_bar_config(disable=True)

    # Setup EggRoll with Implicit Low-Rank Perturbation
    lora_rank = getattr(config, "eggroll_rank", 4)
    # Target modules for SD3
    target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
    
    sigma = getattr(config, "eggroll_sigma", 1e-3)
    lr = getattr(config.train, "learning_rate", 1e-4)
    noise_reuse = getattr(config, "eggroll_noise_reuse", 1)
    
    # Initialize EggRoll (This replaces Linear layers with EggrollLinear)
    # Note: We pass the transformer model directly
    eggroll = EggRoll(
        model=pipeline.transformer,
        target_modules=target_modules,
        sigma=sigma,
        lr=lr,
        group_size=0, # Set group_size=0 for Global Normalization (Magnitude-aware updates)
        noise_reuse=noise_reuse,
        rank=lora_rank,
    )

    # Collect trainable parameters (The weights of the replaced modules)
    # We must ensure they require grad for the optimizer to work, even though we manually set .grad
    trainable_params = []
    for m in eggroll.replaced_modules:
        m.weight.requires_grad_(True)
        trainable_params.append(m.weight)
    
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. EggRoll setup might be incorrect.")

    print(f"EggRoll initialized. Optimizing {len(trainable_params)} tensors (Implicit Rank {lora_rank}).")

    ema = None
    if getattr(config.train, "ema", False):
        # Initialize EMA for trainable parameters
        # decay and update_step_interval are typical values, can be parameterized if needed
        ema = EMAModuleWrapper(trainable_params, decay=0.9999, update_step_interval=1, device=device)
        print("EMA initialized.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_root = config.dataset
    train_dataset = TextPromptDataset(dataset_root, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.sample.train_batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=TextPromptDataset.collate_fn,
    )
    train_loader = accelerator.prepare(train_loader)
    
    # Create test loader for evaluation
    test_dataset = TextPromptDataset(dataset_root, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=TextPromptDataset.collate_fn,
    )
    test_loader = accelerator.prepare(test_loader)

    reward_fn = getattr(flow_rewards, "multi_score")(device, config.reward_fn)

    num_epochs = getattr(config, "num_epochs", 1)
    batches_per_epoch = config.sample.num_batches_per_epoch
    population_size = batches_per_epoch * config.sample.train_batch_size
    
    # Ensure population size is even for antithetic sampling
    if population_size % 2 != 0:
        population_size += 1
        print(f"Adjusted population size to {population_size} to be even for antithetic sampling.")

    global_step = 0
    for epoch in range(num_epochs):
        fitnesses = torch.zeros(population_size, device=device, dtype=torch.float32)
        iterinfos = torch.zeros(population_size, 2, device=device, dtype=torch.int32)

        sample_idx = 0
        loader_iter = iter(train_loader)

        # Training loop
        # We need to process 'population_size' samples
        
        current_pop_processed = 0
        
        # Use tqdm for progress tracking
        # We process 2 samples per iteration (Antithetic), so total steps = population_size
        # Display GLOBAL TOTAL in progress bar for clarity
        global_population_size = population_size * accelerator.num_processes
        pbar = tqdm(total=global_population_size, desc=f"Epoch {epoch} Progress", unit="img", disable=not accelerator.is_main_process)
        
        while current_pop_processed < population_size:
            try:
                prompts, metadatas = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                prompts, metadatas = next(loader_iter)
            
            # Process each prompt in the batch individually to apply unique noise
            for local_idx, prompt in enumerate(prompts):
                if current_pop_processed >= population_size:
                    break
                
                # Check if we have space for the pair
                if sample_idx + 1 >= population_size:
                    break

                # Generate a shared seed for this prompt pair
                # We use a large random integer derived from system time or random generator
                # to ensure randomness across epochs and steps.
                current_seed = random.randint(0, 2**32 - 1)
                
                # We perform Antithetic Sampling: 2 samples per prompt
                # 1. Positive (+Sigma) -> Even ID
                # 2. Negative (-Sigma) -> Odd ID
                
                # Base global index for this process
                # Ensure global offset maintains even/odd parity for multiple GPUs?
                # accelerator.process_index * population_size assumes population_size is even (guaranteed above)
                global_sample_base = sample_idx + accelerator.process_index * population_size
                
                # === Loop for Pair (Positive, Negative) ===
                for i in range(2):
                    current_local_idx = sample_idx + i
                    current_global_idx = global_sample_base + i
                    
                    # iterinfo: [epoch, thread_id]
                    iterinfo = torch.tensor([epoch, current_global_idx], device=device, dtype=torch.int32)
                    
                    # 1. Perturb parameters
                    eggroll.perturb(iterinfo)
                    
                    # 2. Inference
                    # CRITICAL: Use SAME generator for both positive and negative to ensure
                    # geometric consistency (start from same latent noise).
                    # We pass 'generator' to pipeline.
                    generator = torch.Generator(device=device).manual_seed(current_seed)
                    
                    with torch.no_grad():
                        images = pipeline(
                            prompt=[prompt],
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            height=config.resolution,
                            width=config.resolution,
                            output_type="pt",
                            generator=generator # Enforce fixed seed
                        ).images

                    # 3. Unperturb parameters
                    eggroll.unperturb(iterinfo)

                    # 4. Compute Reward
                    rewards = reward_fn(images, [prompt], [metadatas[local_idx]], only_strict=True)

                    # Handle tuple rewards
                    if isinstance(rewards, tuple):
                        rewards = rewards[0]

                    if "avg" in rewards:
                        fitness_value = float(rewards["avg"][0])
                    else:
                        key = list(rewards.keys())[0]
                        fitness_value = float(rewards[key][0])

                    fitnesses[current_local_idx] = fitness_value
                    iterinfos[current_local_idx] = iterinfo

                    if accelerator.is_main_process and i == 0:
                        # Log the first of the pair
                        last_train_sample = (images[0].detach().cpu().clone(), prompts[0], fitness_value)

                # Increment counters by 2 (Pair processed)
                sample_idx += 2
                current_pop_processed += 2
                
                # Update progress bar
                if accelerator.is_main_process:
                    # Scale local progress to global progress for display
                    num_gpus = accelerator.num_processes
                    global_processed = current_pop_processed * num_gpus
                    global_total = population_size * num_gpus
                    
                    pbar.update(2 * num_gpus)
                    pbar.set_description(f"Epoch {epoch} | Pair {global_processed//2}/{global_total//2} | Img {global_processed}/{global_total}")


        pbar.close()


        # 5. Optimization Step
        # Gather all fitnesses and iterinfos from all GPUs
        all_fitnesses = accelerator.gather(fitnesses)
        all_iterinfos = accelerator.gather(iterinfos)
        
        # Log training metrics
        if config.use_wandb and accelerator.is_main_process:
             # 1. 평균 점수
             mean_fitness = all_fitnesses.mean().item()
             
             # 2. [NEW] Pairwise Difference (순수 노이즈 영향력)
             # 배열을 짝수/홀수 인덱스로 나눕니다.
             # shape가 (N,) 이라고 가정합니다.
             pos_scores = all_fitnesses[0::2] # 짝수 인덱스 (+Sigma)
             neg_scores = all_fitnesses[1::2] # 홀수 인덱스 (-Sigma)
             
             # 두 점수 차이의 절대값 평균
             # 이 값이 너무 크면 노이즈 과다, 0에 가까우면 노이즈 부족
             noise_impact = torch.abs(pos_scores - neg_scores).mean().item()
             
             # 3. Pairwise Difference의 표준편차 (안정성 확인용)
             # 모든 프롬프트에서 노이즈 영향이 균일한지 봅니다.
             noise_impact_std = torch.abs(pos_scores - neg_scores).std().item()

        
        # Apply Rank-based Fitness Shaping with Temperature
        # This transforms raw scores into robust ranks and then applies softmax sharpening
        # providing aggressive yet stable gradient pressure.
        temperature = getattr(config, "eggroll_temperature", 0.1)
        shaped_fitnesses = compute_softmax_ranks(all_fitnesses, temperature=temperature)
        
        eggroll_metrics = eggroll.step(shaped_fitnesses, all_iterinfos) # Updates base weights via standard optimizer inside
        
        if config.use_wandb and accelerator.is_main_process:
             # Combine original reward metrics and EggRoll training metrics
             log_data = {
                 "train/reward_avg": mean_fitness,
                 "train/noise_impact": noise_impact,
                 "train/noise_impact_std": noise_impact_std,
                 "train/grad_norm": eggroll_metrics["grad_norm"],
                 "train/update_param_ratio": eggroll_metrics["update_param_ratio"],
                 "train/grad_cosine_sim": eggroll_metrics["grad_cosine_sim"],
                 "train/winner_alignment": eggroll_metrics["winner_alignment"],
                 "epoch": epoch,
             }
             wandb.log(log_data, step=epoch)
        accelerator.wait_for_everyone()
        
        # Update EMA
        if ema is not None:
             # EggRoll step effectively performs one optimization step.
             # We increment global_step here to track optimization steps.
             global_step += 1
             ema.step(trainable_params, global_step)

        if accelerator.is_main_process:
            print(f"Epoch {epoch} step completed. Mean fitness: {all_fitnesses.mean().item()}")

             # 6. Save Checkpoint
            save_freq = getattr(config, "save_freq", 1)
            if (epoch + 1) % save_freq == 0:
                 if ema is not None:
                     ema.copy_ema_to(trainable_params, store_temp=True)
                 
                 save_dir = getattr(config, "save_dir", config.logdir)
                 # Use run_name (which includes timestamp) in the path
                 # Structure: log_dir/run_name/checkpoints/checkpoint-epoch-N/lora
                 save_path = os.path.join(save_dir, config.run_name, "checkpoints", f"checkpoint-epoch-{epoch+1}", "lora")
                 os.makedirs(save_path, exist_ok=True)
                 pipeline.transformer.save_pretrained(save_path)
                 print(f"Saved checkpoint to {save_path}")
                 
                 if ema is not None:
                     ema.copy_temp_to(trainable_params)

        # Evaluation Loop
        eval_freq = getattr(config, "eval_freq", 20)
        if (epoch + 1) % eval_freq == 0:
             print(f"Running evaluation at epoch {epoch+1}...")
             
             if ema is not None:
                 ema.copy_ema_to(trainable_params, store_temp=True)

             pipeline.transformer.eval() # Ensure eval mode (though we froze base model, safety check)
             
             all_eval_rewards = defaultdict(list)
             eval_images_log = []
             
             # Limit eval batches to avoid taking too long if dataset huge
             # or just run full test set if small. Using logic similar to og script
             
             for i, (prompts, metadatas) in enumerate(tqdm(test_loader, desc="Evaluation", disable=not accelerator.is_main_process)):
                 with torch.no_grad():
                     # Encode prompts
                     # Create negative prompts (empty string) to match OG behavior
                     negative_prompts = [""] * len(prompts)
                     
                     # Simple inference
                     generator = torch.Generator(device=device).manual_seed(42 + i)
                     
                     images = pipeline(
                        prompt=prompts,
                        negative_prompt=negative_prompts,
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        height=config.resolution,
                        width=config.resolution,
                        output_type="pt",
                        generator=generator,
                     ).images
                 
                 # Compute rewards
                 rewards = reward_fn(images, prompts, metadatas, only_strict=True)
                 if isinstance(rewards, tuple):
                     rewards = rewards[0]
                
                 # Gather rewards
                 for key, value in rewards.items():
                     # value might be a list or tensor. Ensure it is a tensor.
                     if isinstance(value, list):
                         value = torch.tensor(value, device=device)
                     
                     # value is tensor of shape (batch_size,)
                     gathered_val = accelerator.gather(value.to(device).float()).cpu().numpy()
                     all_eval_rewards[key].append(gathered_val)
                 
                 # Gather images for logging (only first batch or few samples)
                 if i == 0:
                      # Directly use LOCAL images and prompts from the main process.
                      # This guarantees alignment because they come from the same pipeline call.
                      if accelerator.is_main_process:
                           # Cast to float32 numpy
                           local_images_np = images.float().cpu().numpy()
                           
                           # Limit to first 4 images
                           num_log_images = min(4, len(local_images_np))
                           
                           with tempfile.TemporaryDirectory() as tmpdir:
                               for idx in range(num_log_images):
                                   img = local_images_np[idx]
                                   pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
                                   pil = pil.resize((config.resolution, config.resolution))
                                   
                                   # Strictly use LOCAL prompts
                                   current_prompt = prompts[idx]
                                   
                                   # Get reward string
                                   r_str = ""
                                   for r_k, r_v in rewards.items():
                                        # rewards dict values are tensors of shape (batch_size,) or lists
                                        # We need the value at [idx]
                                        if idx < len(r_v):
                                            val = r_v[idx]
                                            # If tensor, retrieve item. If float/int, use directly.
                                            val_scalar = val.item() if hasattr(val, 'item') else val
                                            r_str += f"{r_k}: {val_scalar:.2f} "
                                   
                                   # Pass PIL image directly to wandb to avoid file path issues with tempdirs
                                   eval_images_log.append(wandb.Image(pil, caption=f"{current_prompt[:100]}... | {r_str}"))

                 # Explicitly delete tensors to free memory
                 del images
                 torch.cuda.empty_cache()

             if config.use_wandb and accelerator.is_main_process:
                 log_dict = {"epoch": epoch}
                 if eval_images_log:
                     log_dict["eval_images"] = eval_images_log
                 
                 for key, val_list in all_eval_rewards.items():
                     # Concatenate all batches
                     full_arr = np.concatenate(val_list)
                     log_dict[f"eval_reward_{key}"] = np.mean(full_arr)
                 
                 wandb.log(log_dict, step=epoch)
             
             if ema is not None:
                 ema.copy_temp_to(trainable_params)

def main(_):
    config = FLAGS.config
    train_sd3_eggroll(config)

if __name__ == "__main__":
    app.run(main)
