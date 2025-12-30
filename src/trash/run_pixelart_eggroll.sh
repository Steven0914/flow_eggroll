PYTHONPATH=. accelerate launch --config_file flow_grpo/scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 src/train_pixelart.py --config flow_grpo/config/grpo.py:pixelart_sigma_512
