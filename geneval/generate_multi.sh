#!/bin/bash

# 设置基础变量
PROMPT_PATH=/workspace/sii-jqy/geneval/prompts/val_prompts.jsonl
MODEL_PATH="/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146/stable-diffusion-3.5-medium"
workspace="/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146"
OUT_BASE="/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146/geneval/experiments"

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_PROCESS=4

# 从2000到60000，每2000一个间隔
for ckpt in $(seq 2000 2000 18000); do
    echo "=========================================="
    echo "Testing checkpoint: $ckpt"
    echo "=========================================="
    
    # 设置输出路径
    OUT_PATH="$OUT_BASE/ckpt-$ckpt"
    
    # 删除原有模型文件
    # rm -f $workspace/stable-diffusion-3.5-medium/transformer/diffusion_pytorch_model.safetensors
    
    # # 复制对应的checkpoint
    # cp $workspace/diffusers/experiments/checkpoint-$ckpt/transformer/diffusion_pytorch_model.safetensors $workspace/stable-diffusion-3.5-medium/transformer/
    
    # # 创建输出目录
    # mkdir -p $OUT_PATH
    
    # # 生成图片
    # echo "Generating images..."
    # accelerate launch --main_process_port 29502 --num_processes=$NUM_PROCESS generation/diffusers_generate2.py \
    #     $PROMPT_PATH \
    #     --model $MODEL_PATH \
    #     --outdir $OUT_PATH \
    #     --steps 50 \
    #     --scale 4.5 \
    #     --H 512 \
    #     --W 512
    
    
    # 评估图片
    echo "Evaluating images..."
    python evaluation/evaluate_images.py \
        $OUT_PATH \
        --outfile $OUT_PATH/results.jsonl \
        --model-path models
    
    # 汇总分数
    echo "Summarizing scores..."
    python evaluation/summary_scores.py \
        $OUT_PATH/results.jsonl > $OUT_PATH/scores.txt
    
    # 显示结果
    echo "Results for checkpoint $ckpt:"
    cat $OUT_PATH/scores.txt
    echo ""
    
done

echo "All checkpoints tested!"