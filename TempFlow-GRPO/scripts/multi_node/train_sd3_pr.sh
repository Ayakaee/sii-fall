#!/bin/bash
# Common part for all nodes 
export PYTHONPATH="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/TempFlow-GRPO:$PYTHONPATH"
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/project/25jinqiu14/sunyihang-P
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
--num_processes=4 \
--main_process_port 29501 \
    scripts/train_sd3_pr.py \
    --config config/dgx.py:geneval_sd3_pr

