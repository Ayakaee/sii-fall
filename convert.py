import pandas as pd
import datasets
import json
import re
from tqdm import tqdm
import os
from PIL import Image

# --- 配置 ---
# 输入1：您的原始Arrow数据集目录 (包含Pillow图像对象)
ORIGINAL_DATASET_DIR = '/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin/data/train/'

# 输入2：包含成功改写的JSONL文件
SUCCESSFUL_JSONL_PATH = 'intermediate_successful.jsonl' 

# 输出：最终的、可直接用于训练的Arrow数据集目录
FINAL_ARROW_DIR = '/inspire/ssd/project/25jinqiu14/sunyihang-P-253130146/data_new'

# 日志：记录JSONL中无法修复的行
ERROR_LOG_PATH = 'final_assembly_errors.log'                   

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
    # --- 步骤 1: 加载并准备成功改写的数据 ---
    print(f"--- Step 1: Loading successful rewrites from '{SUCCESSFUL_JSONL_PATH}' ---")
    if not os.path.exists(SUCCESSFUL_JSONL_PATH):
        print(f"Error: Successful rewrites file not found at '{SUCCESSFUL_JSONL_PATH}'. Exiting.")
        exit()

    successful_records = [repair_and_parse_json_line(line) for line in open(SUCCESSFUL_JSONL_PATH, 'r', encoding='utf-8')]
    successful_records = [r for r in successful_records if r is not None] # 过滤掉解析失败的None
            
    df_rewrites = pd.DataFrame(successful_records)
    print(f"Loaded {len(df_rewrites)} successful rewrite records.")

    # 从 'file_name' (e.g., 'xyz.jpg') 中提取 'image_id' ('xyz')
    df_rewrites['image_id'] = df_rewrites['file_name'].str.replace(r'\..*$', '', regex=True)
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
    print("\n--- Step 3: Merging data, updating captions, and cleaning up ---")
    
    # 确保合并键的数据类型一致，这是稳健操作的关键
    df_original['image_id'] = df_original['image_id'].astype(str)
    df_rewrites['image_id'] = df_rewrites['image_id'].astype(str)

    # 使用内连接(inner merge)来自动筛选出共同存在的条目
    df_merged = pd.merge(df_original, df_rewrites, on='image_id', how='inner')
    
    print(f"Merge complete. Found {len(df_merged)} matching records to update.")
    
    # 核心操作：用新的caption替换旧的
    df_merged['caption'] = df_merged['new_caption']
    
    # 清理：只保留原始数据集的列，确保schema完全一致
    final_columns = original_dataset.column_names
    df_final = df_merged[final_columns]

    print("Caption column updated. Final data is ready.")

    # --- 步骤 4: 保存为最终的 Arrow 数据集 (保留原始schema) ---
    print("\n--- Step 4: Saving final Arrow dataset with original schema ---")
    
    # 从Pandas DataFrame转换回Hugging Face Dataset对象
    # 【关键】我们传递原始的features，以确保Image对象等特殊类型被正确保留
    final_dataset = datasets.Dataset.from_pandas(df_final, features=original_dataset.features, preserve_index=False)
    
    if os.path.exists(FINAL_ARROW_DIR):
        print(f"Warning: Output directory '{FINAL_ARROW_DIR}' already exists. It will be overwritten.")
    
    final_dataset.save_to_disk(FINAL_ARROW_DIR)
    
    print(f"\nSuccessfully saved final Arrow dataset with {len(final_dataset)} items to '{FINAL_ARROW_DIR}'")
    
    # --- 步骤 5: 最终验证 ---
    print("\n--- Step 5: Final verification ---")
    print("Schema of the new dataset (should match original):")
    print(final_dataset.info)
    
    print("\nVerifying one record...")
    first_record = final_dataset[0]
    image_object = first_record['image']
    print(f"  - Caption has been updated to: '{first_record['caption'][:100]}...'")
    print(f"  - 'image' column is a '{type(image_object)}' object.")
    if isinstance(image_object, Image.Image):
        print("  - Verification successful! Image is a Pillow object.")
    else:
        print("  - Verification FAILED! Image is NOT a Pillow object.")

    print("\nProcess complete!")