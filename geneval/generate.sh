PROMPT_PATH=/workspace/sii-jqy/geneval/prompts/val_prompts.jsonl
MODEL_PARH="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/stable-diffusion-3.5-medium"
workspace="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146"
ckpt=6000
OUT_PATH="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/geneval/experiments/ckpt-22k"

rm $workspace/stable-diffusion-3.5-medium/transformer/diffusion_pytorch_model.safetensors
cp $workspace/diffusers/experiments/checkpoint-$ckpt/transformer/diffusion_pytorch_model.safetensors $workspace/stable-diffusion-3.5-medium/transformer/

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

python evaluation/evaluate_images.py \
    $OUT_PATH \
    --outfile $OUT_PATH/results.jsonl \
    --model-path models \

python evaluation/summary_scores.py \
$OUT_PATH/results.jsonl > $OUT_PATH/scores.txt


cat $OUT_PATH/scores.txt