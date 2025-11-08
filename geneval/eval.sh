export MODEL_NAME="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/stable-diffusion-3.5-medium"
export DATASET_NAME="/inspire/hdd/project/25jinqiu14/public/datasets/vismin"
export OUTPUT_DIR="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/diffusers/experiments"
export OUT_PATH="/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/geneval/experiments"

python evaluation/evaluate_images.py \
    /inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/geneval/images \
    $OUT_PATH \
    --outfile $OUT_PATH/results.jsonl \
    --model-path models \
    


python evaluation/summary_scores.py \
$OUT_PATH/results.jsonl > $OUT_PATH/scores.txt


cat $OUT_PATH/scores.txt