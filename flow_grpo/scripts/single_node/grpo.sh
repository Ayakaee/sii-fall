# 1 GPU
export PYTHONPATH="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/flow_grpo:$PYTHONPATH"
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/flow_grpo

accelerate launch \
--config_file scripts/accelerate_configs/multi_gpu.yaml \
--num_processes=4 \
--main_process_port 29501 \
scripts/train_sd3.py \
--config config/grpo.py:geneval_sd3
# 4 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
