# PYTHONPATH=. accelerate launch --config_file flow_grpo/scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 src/train_sd3.py --config src/config/config.py:general_ocr_sd3_1gpu


PYTHONPATH=. accelerate launch --config_file flow_grpo/scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 src/train_sd3.py --config src/config/config.py:pickscore_sd3