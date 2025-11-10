export MODEL_NAME="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/stable-diffusion-3.5-medium"
export DATASET_NAME="/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146/data_new"
export OUTPUT_DIR="/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146/diffusers/experiments"
export WANDB_MODE=offline
export WANDB_DIR="/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146/diffusers"
export WANDB_NAME="full-sft-with-revised-prompt"

mkdir -p "$OUTPUT_DIR"
export workspace="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146"
export ckpt=6000

rm $workspace/stable-diffusion-3.5-medium/transformer/diffusion_pytorch_model.safetensors
cp $workspace/diffusers/experiments/checkpoint-$ckpt+16k/transformer/diffusion_pytorch_model.safetensors $workspace/stable-diffusion-3.5-medium/transformer/

accelerate launch train_text_to_image_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --caption_column="caption" \
  --validation_prompts "a photo of a zebra above a car" "a photo of a zebra behind a car" "A capybara holding a sign that reads Hello World" "A man with a red jacket and glasses standing by a window" \
  --num_validation_images=2 \
  --validation_steps=2000 \
  --checkpointing_steps=2000 \
  --resolution=512 \
  --max_train_steps=18000 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --seed=42 \
  --report_to wandb \
  --output_dir=$OUTPUT_DIR \
  2>&1 | tee "$OUTPUT_DIR/train.log"

cd /inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/geneval
bash generate_multi.sh