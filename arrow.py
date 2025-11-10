import pandas as pd
import datasets
import json
import re
from tqdm import tqdm
import os

# --- 配置 ---
MODEL_PATH = "Qwen/Qwen3-8B"  
ORIGINAL_DATASET_DIR = '/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin/data/train' # 您的原始Arrow数据集目录

# --- 输出文件路径配置 ---
OUTPUT_DATASET_DIR = '/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146/dataset/' 
ERROR_LOG_PATH = 'merge_errors.log'                   # 输出：记录无法修复的行的日志
SUCCESSFUL_JSONL_PATH = '/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/intermediate_successful.jsonl'
def repair_and_parse_json_line(line):
    """尝试修复并解析一个可能损坏的JSON行，返回一个字典。"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        match = re.search(r'^{"file_name":\s*"(.*?)",\s*"text":\s*"(.*)"}\s*$', line)
        if match:
            return {"file_name": match.group(1), "text": match.group(2)}
    return None

if __name__ == "__main__":
    # --- 步骤 1: 加载并修复成功改写的数据 ---
    print(f"--- Step 1: Loading successful rewrites from '{SUCCESSFUL_JSONL_PATH}' ---")
    if not os.path.exists(SUCCESSFUL_JSONL_PATH):
        print(f"Error: Successful rewrites file not found at '{SUCCESSFUL_JSONL_PATH}'. Exiting.")
        exit()

    successful_records = []
    failed_lines = []
    with open(SUCCESSFUL_JSONL_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Repairing JSONL"):
        record = repair_and_parse_json_line(line)
        if record:
            successful_records.append(record)
        else:
            failed_lines.append(f"Line {i+1}: {line.strip()}\n")
            
    if failed_lines:
        print(f"Warning: {len(failed_lines)} lines in JSONL could not be parsed. See '{ERROR_LOG_PATH}'.")
        with open(ERROR_LOG_PATH, 'w', encoding='utf-8') as f:
            f.writelines(failed_lines)
            
    df_rewrites = pd.DataFrame(successful_records)
    print(f"Loaded {len(df_rewrites)} successful rewrite records.")

    # 准备用于合并的列
    # 从 'file_name' (e.g., 'xyz.jpg') 中提取 'image_id' ('xyz')
    df_rewrites['image_id'] = df_rewrites['file_name'].str.replace(r'\.jpg$', '', regex=True)
    df_rewrites.rename(columns={'text': 'new_caption'}, inplace=True)
    df_rewrites = df_rewrites[['image_id', 'new_caption']]

    # --- 步骤 2: 加载原始Arrow数据集 ---
    print(f"\n--- Step 2: Loading original dataset from '{ORIGINAL_DATASET_DIR}' ---")
    if not os.path.exists(ORIGINAL_DATASET_DIR):
        print(f"Error: Original dataset directory not found at '{ORIGINAL_DATASET_DIR}'. Exiting.")
        exit()
        
    original_dataset = datasets.load_from_disk(ORIGINAL_DATASET_DIR)
    df_original = original_dataset.to_pandas()
    print(f"Loaded {len(df_original)} original records.")

    # --- 步骤 3: 合并、替换、清理 ---
    print("\n--- Step 3: Merging, updating, and cleaning data ---")
    
    # 为确保合并成功，将两个DataFrame的 'image_id' 列都转换为字符串类型
    df_original['image_id'] = df_original['image_id'].astype(str)
    df_rewrites['image_id'] = df_rewrites['image_id'].astype(str)

    # 使用内连接(inner merge)来筛选并合并数据
    # 这会保留 df_original 的所有列，并添加 df_rewrites 的列
    df_merged = pd.merge(df_original, df_rewrites, on='image_id', how='inner')
    
    print(f"Merge complete. Found {len(df_merged)} matching records.")
    
    # 核心操作：用新的caption替换旧的
    df_merged['caption'] = df_merged['new_caption']
    
    # 清理：只保留原始数据集的列，以保持schema一致
    final_columns = original_dataset.column_names
    df_final = df_merged[final_columns]

    print("Caption column updated successfully.")

    # --- 步骤 4: 保存为新的 Arrow 数据集 ---
    print("\n--- Step 4: Saving final Arrow dataset ---")
    
    final_dataset = datasets.Dataset.from_pandas(df_final, preserve_index=False)
    
    if os.path.exists(OUTPUT_DATASET_DIR):
        print(f"Warning: Output directory '{OUTPUT_DATASET_DIR}' already exists. It will be overwritten.")
    
    final_dataset.save_to_disk(OUTPUT_DATASET_DIR)
    
    print(f"\nSuccessfully saved final Arrow dataset with {len(final_dataset)} items to '{OUTPUT_DATASET_DIR}'")
    print("Schema of the new dataset:")
    print(final_dataset.info)
    print("\nProcess complete!")