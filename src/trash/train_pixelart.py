import os
import sys
from typing import List
from absl import app, flags
from ml_collections import config_flags
from typing import List
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow_grpo"))
import flow_grpo.rewards as flow_rewards

from noiser.eggroll import EggRoll

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")


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


def train_pixelart_eggroll(config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PixArt
    if "512" in config.pretrained.model and "PixArt" in config.pretrained.model:
        transformer = PixArtTransformer2DModel.from_pretrained(config.pretrained.model, subfolder="transformer", torch_dtype=torch.bfloat16)
        pipeline = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", transformer=transformer, torch_dtype=torch.bfloat16)
    else:
        pipeline = PixArtSigmaPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.bfloat16)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False) # Freeze base model

    pipeline.to(device)
    pipeline.safety_checker = None

    # Setup LoRA using PEFT
    lora_rank = getattr(config, "eggroll_rank", 4)
    # Target modules as seen in train_sd3.py
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=getattr(config, "lora_alpha", 64), # Default from train_sd3.py
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    
    # Wrap transformer with PEFT
    pipeline.transformer = get_peft_model(pipeline.transformer, peft_config)
    pipeline.transformer.print_trainable_parameters()

    # Enable model CPU offload
    # pipeline.enable_model_cpu_offload()
    
    # Collect trainable parameters (LoRA weights)
    trainable_params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. LoRA setup might be incorrect.")

    sigma = getattr(config, "eggroll_sigma", 1e-3)
    lr = getattr(config.train, "learning_rate", 1e-4)
    noise_reuse = getattr(config, "eggroll_noise_reuse", 1)
    
    # Initialize EggRoll
    # Note: we don't need 'group_size' or 'freeze_nonlora' as explicit args as we pass filtered params
    eggroll = EggRoll(trainable_params, sigma=sigma, lr=lr, group_size=0, noise_reuse=noise_reuse)

    dataset_root = config.dataset
    train_dataset = TextPromptDataset(dataset_root, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.sample.train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=TextPromptDataset.collate_fn,
    )

    reward_fn = getattr(flow_rewards, "multi_score")(device, config.reward_fn)

    num_epochs = getattr(config, "num_epochs", 1)
    batches_per_epoch = config.sample.num_batches_per_epoch
    population_size = batches_per_epoch * config.sample.train_batch_size

    for epoch in range(num_epochs):
        fitnesses = torch.zeros(population_size, device=device, dtype=torch.float32)
        iterinfos = torch.zeros(population_size, 2, device=device, dtype=torch.int32)

        sample_idx = 0
        loader_iter = iter(train_loader)

        # Training loop
        # We need to process 'population_size' samples
        
        current_pop_processed = 0
        
        # Use tqdm for progress tracking
        pbar = tqdm(total=population_size, desc=f"Epoch {epoch} Sampling")
        
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
                iterinfo = torch.tensor([epoch, sample_idx], device=device, dtype=torch.int32)
                
                # 1. Perturb parameters (Apply Noise)
                eggroll.perturb(iterinfo)
                
                # 2. Inference
                with torch.no_grad():
                    added_cond_kwargs = {
                        "resolution": torch.tensor([config.resolution, config.resolution], device=device).repeat(1, 1),
                        "aspect_ratio": torch.tensor([1.0], device=device).repeat(1, 1)
                    }
                    images = pipeline(
                        prompt=[prompt],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        height=config.resolution,
                        width=config.resolution,
                        output_type="pt",
                        added_cond_kwargs=added_cond_kwargs,
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

                sample_idx += 1
                current_pop_processed += 1
                pbar.update(1)
        
        pbar.close()

        # 5. Optimization Step
        eggroll.step(fitnesses, iterinfos) # Updates base weights via standard optimizer inside
        print(f"Epoch {epoch} step completed. Mean fitness: {fitnesses.mean().item()}")

        # 6. Save Checkpoint
        save_freq = getattr(config, "save_freq", 1)
        if (epoch + 1) % save_freq == 0:
             save_dir = config.logdir
             # Use epoch as step identifier
             save_path = os.path.join(save_dir, "checkpoints", f"checkpoint-epoch-{epoch+1}", "lora")
             os.makedirs(save_path, exist_ok=True)
             pipeline.transformer.save_pretrained(save_path)
             print(f"Saved checkpoint to {save_path}")

def main(_):
    config = FLAGS.config
    train_pixelart_eggroll(config)

if __name__ == "__main__":
    app.run(main)
