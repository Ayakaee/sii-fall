export MODEL_NAME="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/stable-diffusion-3.5-medium"
export DATASET_NAME="/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin"
export OUTPUT_DIR="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/diffusers/experiments_lora"
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/diffusers

mkdir -p "$OUTPUT_DIR"

accelerate launch train_text_to_image_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --caption_column="caption" \
  --validation_prompt="a photo of a zebra above a car" \
  --num_validation_images=4 \
  --validation_epochs=5 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --resolution=512 \
  --max_train_steps=10000 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --text_encoder_lr=5e-5 \
  --rank=4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --report_to=wandb \
  --output_dir=$OUTPUT_DIR \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  2>&1 | tee "$OUTPUT_DIR/train.log"

