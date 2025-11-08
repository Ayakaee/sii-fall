export MODEL_NAME="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/stable-diffusion-3.5-medium"
export DATASET_NAME="/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin"
export OUTPUT_DIR="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/diffusers/experiments"
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/diffusers

mkdir -p "$OUTPUT_DIR"

accelerate launch train_text_to_image_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --caption_column="caption" \
  --validation_prompts "a photo of a zebra above a car" "a photo of a zebra behind a car" "A capybara holding a sign that reads Hello World" "A man with a red jacket and glasses standing by a window" \
  --num_validation_images=2 \
  --validation_steps=2000 \
  --checkpointing_steps=2000 \
  --resolution=512 \
  --max_train_steps=15000 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --seed=42 \
  --report_to wandb \
  --output_dir=$OUTPUT_DIR \
  2>&1 | tee "$OUTPUT_DIR/train.log"