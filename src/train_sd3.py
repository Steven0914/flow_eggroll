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
        group_size=0,
        noise_reuse=noise_reuse,
        rank=lora_rank
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
        pbar = tqdm(total=population_size, desc=f"Epoch {epoch} Sampling", disable=not accelerator.is_main_process)
        
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
                
                # Correct sample index for noise generation
                # In original code: sample_idx tracked total samples in epoch
                global_sample_idx = sample_idx + accelerator.process_index * population_size
                iterinfo = torch.tensor([epoch, global_sample_idx], device=device, dtype=torch.int32)
                
                # 1. Perturb parameters (Apply Noise)
                eggroll.perturb(iterinfo)
                
                # 2. Inference
                with torch.no_grad():
                    images = pipeline(
                        prompt=[prompt],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        height=config.resolution,
                        width=config.resolution,
                        output_type="pt",
                    ).images

                # 3. Unperturb parameters (Restore Base Weights)
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

                fitnesses[sample_idx] = fitness_value
                iterinfos[sample_idx] = iterinfo

                if accelerator.is_main_process:
                    # Store sample for logging (cpu move to save gpu mem)
                    # Use clone to ensure it persists safely if tensor memory is reused/freed
                    # Store tuple: (Image Tensor, Prompt String, Reward Score)
                    last_train_sample = (images[0].detach().cpu().clone(), prompts[0], fitness_value)

                sample_idx += 1
                current_pop_processed += 1
                pbar.update(1)

        pbar.close()

        # Log training sample images every N epochs
        # if config.use_wandb and accelerator.is_main_process and (epoch + 1) % 1 == 0:
        #     if last_train_sample is not None:
        #         # Unpack safely
        #         img_tensor, p_text, r_score = last_train_sample
                
        #         # Cast to float32 before numpy conversion (it is already on cpu, but just to be safe with types)
        #         pil = Image.fromarray(
        #             (img_tensor.float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        #         )
        #         pil = pil.resize((config.resolution, config.resolution))
                
        #         wandb.log({
        #             "images": [wandb.Image(pil, caption=f"Epoch {epoch} | {p_text[:100]} | Score: {r_score:.2f}")]
        #         }, step=epoch)

        # 5. Optimization Step
        # Gather all fitnesses and iterinfos from all GPUs
        all_fitnesses = accelerator.gather(fitnesses)
        all_iterinfos = accelerator.gather(iterinfos)
        
        # Log training metrics
        if config.use_wandb and accelerator.is_main_process:
             mean_fitness = all_fitnesses.mean().item()
             wandb.log({
                 "epoch": epoch,
                 "reward_avg": mean_fitness, # Logging avg reward
             }, step=epoch)

        eggroll.step(all_fitnesses, all_iterinfos) # Updates base weights via standard optimizer inside
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
                     images = pipeline(
                        prompt=prompts,
                        negative_prompt=negative_prompts,
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        height=config.resolution,
                        width=config.resolution,
                        output_type="pt",
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
                                   eval_images_log.append(wandb.Image(pil, caption=f"{current_prompt[:50]}... | {r_str}"))

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
