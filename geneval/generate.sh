MODEL_PARH=/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/stable-diffusion-3.5-medium
PROMPT_PATH=/workspace/sii-jqy/geneval/prompts/val_prompts.jsonl
OUT_PATH=images
mkdir -p $OUT_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_PROCESS=4
accelerate launch --num_processes=$NUM_PROCESS generation/diffusers_generate.py \
    $PROMPT_PATH \
    --model $MODEL_PARH \
    --outdir $OUT_PATH \
    --steps 50 \
    --scale 4.5 \
    --H 512 \
    --W 512