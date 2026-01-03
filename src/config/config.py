import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "flow_grpo/dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    
    config.train.learning_rate = 5e-3

    # eggroll
    config.eggroll_sigma = 1e-2
    config.eggroll_temperature = 0.5

    return config



def pickscore_sd3_2gpu():
    gpu_number=2
    config = compressibility()
    config.use_wandb = True

    config.dataset = os.path.join(os.getcwd(), "flow_grpo/dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 2

    # Total population size
    config.population_size = 256
    
    # Calculate batches per epoch
    total_batch_size_per_step = gpu_number * config.sample.train_batch_size
    config.sample.num_batches_per_epoch = config.population_size // total_batch_size_per_step
    
    assert config.population_size % total_batch_size_per_step == 0, f"Population size {config.population_size} must be divisible by total batch size ({total_batch_size_per_step})"
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please ensure num_batches_per_epoch is even."
    

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.test_batch_size = 4

    # eggroll
    config.eggroll_sigma = 1e-2
    config.save_freq = 100 # epoch
    config.eval_freq = 20
    
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.eggroll_temperature = 0.5
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3():
    gpu_number=1
    config = compressibility()
    config.use_wandb = True

    config.dataset = os.path.join(os.getcwd(), "flow_grpo/dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 2

    # Total population size
    config.population_size = 256
    
    # Calculate batches per epoch
    total_batch_size_per_step = gpu_number * config.sample.train_batch_size
    config.sample.num_batches_per_epoch = config.population_size // total_batch_size_per_step
    
    assert config.population_size % total_batch_size_per_step == 0, f"Population size {config.population_size} must be divisible by total batch size ({total_batch_size_per_step})"
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please ensure num_batches_per_epoch is even."
    

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.test_batch_size = 2

    # eggroll
    config.eggroll_sigma = 1e-2
    config.save_freq = 100 # epoch
    config.eval_freq = 5
    config.eggroll_temperature = 0.5
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_2gpu():
    gpu_number = 2
    config = compressibility()
    config.use_wandb = True
    config.dataset = os.path.join(os.getcwd(), "flow_grpo/dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    
    # Total population size
    config.population_size = 2048
    
    # Calculate batches per epoch
    total_batch_size_per_step = gpu_number * config.sample.train_batch_size
    config.sample.num_batches_per_epoch = config.population_size // total_batch_size_per_step
    
    assert config.population_size % total_batch_size_per_step == 0, f"Population size {config.population_size} must be divisible by total batch size ({total_batch_size_per_step})"
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please ensure num_batches_per_epoch is even."

    config.sample.test_batch_size = 8 

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True

    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 20 # epoch
    config.eval_freq = 5
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def general_ocr_sd3_1gpu():
    gpu_number = 1
    config = compressibility()
    config.use_wandb = True
    config.dataset = os.path.join(os.getcwd(), "flow_grpo/dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    
    # Total population size
    config.population_size = 512
    
    # Calculate batches per epoch
    total_batch_size_per_step = gpu_number * config.sample.train_batch_size
    config.sample.num_batches_per_epoch = config.population_size // total_batch_size_per_step
    
    assert config.population_size % total_batch_size_per_step == 0, f"Population size {config.population_size} must be divisible by total batch size ({total_batch_size_per_step})"
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please ensure num_batches_per_epoch is even."

    config.sample.test_batch_size = 8 

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True

    config.save_freq = 10 # epoch
    config.eval_freq = 5
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def general_ocr_sd3_2gpu_test():
    gpu_number = 2
    config = compressibility()
    config.use_wandb = True
    config.dataset = os.path.join(os.getcwd(), "flow_grpo/dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    
    # Total population size
    config.population_size = 64
    
    # Calculate batches per epoch
    total_batch_size_per_step = gpu_number * config.sample.train_batch_size
    config.sample.num_batches_per_epoch = config.population_size // total_batch_size_per_step
    
    assert config.population_size % total_batch_size_per_step == 0, f"Population size {config.population_size} must be divisible by total batch size ({total_batch_size_per_step})"
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please ensure num_batches_per_epoch is even."

    config.sample.test_batch_size = 8

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99

    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.eggroll_rank = 16

    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 10 # epoch
    config.eval_freq = 10
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config




def get_config(name):
    return globals()[name]()
