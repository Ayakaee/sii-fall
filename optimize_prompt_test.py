import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import time
import re

# --- 1. Configuration ---
MODEL_PATH = "Qwen/Qwen3-8B" 
INPUT_CSV_PATH = 'dataset_preview/metadata.csv'          # CHANGE THIS to your input file

# --- 输出文件路径配置 ---
SUCCESS_JSONL_PATH = 'training_data_final.jsonl' # 成功的训练文件
GROUP_CSV_PATH = 'filtered_groups_log.csv'       # 因'group'描述被过滤的文件
FAILED_TXT_PATH = 'failed_rewrites_ids.txt'      # 转换失败的 image_id 列表
BATCH_SIZE = 64

# --- 2. Load Model & Tokenizer (不变) ---
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully on device:", model.device)
print("Configuring padding token...")
pad_token_id = tokenizer.eos_token_id
if isinstance(pad_token_id, list):
    pad_token_id = pad_token_id[0]
tokenizer.pad_token_id = pad_token_id
model.generation_config.pad_token_id = pad_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.decode(pad_token_id)
print(f"Padding token ID successfully set to: {tokenizer.pad_token_id}")

# --- 3. LogitsProcessor (不变) ---
class SuppressThinkProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.forbidden_token_id = tokenizer.encode("<|think|>", add_special_tokens=False)[0]
    def __call__(self, input_ids, scores):
        scores[:, self.forbidden_token_id] = -float("inf")
        return scores
logits_processor_list = LogitsProcessorList([SuppressThinkProcessor(tokenizer)])

# --- 4. Meta-Prompt (不变) ---
META_PROMPT_CONTENT = """You are a highly specialized data transformation bot. Your single function is to convert descriptive sentences into precise, non-creative, text-to-image commands.
You must follow these non-negotiable rules:
1. Your ENTIRE response MUST be the final command itself.
2. The command MUST start with "a photo of ".
3. There must be NO explanations, NO conversation, NO apologies, and NO additional text whatsoever.
Adhere strictly to the format shown in the examples.
--- EXAMPLES ---
Input: "A man with a red jacket and glasses standing by a window."
Output: a photo of a man with a red jacket and glasses
Input: "A gray cat is laying on top of a black car"
Output: a photo of a gray cat on top of a black car
"""
def create_qwen_prompt(caption):
    user_content = f"{META_PROMPT_CONTENT}\n\n--- TASK ---\nInput: \"{caption}\"\nOutput:"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- 5. Batch Rewriting Function (不变) ---
def rewrite_batch(captions):
    prompts = [create_qwen_prompt(cap) for cap in captions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []
    for output in decoded_outputs:
        try:
            assistant_response = output.split('assistant\n')[-1].strip()
            # 即使有<think>标签，我们的解析逻辑依然稳健
            if '</think>' in assistant_response:
                final_answer = assistant_response.split('</think>')[-1].strip()
            else:
                final_answer = assistant_response
            results.append(final_answer)
        except Exception:
            results.append("REWRITE_FAILED_PARSING")
            
    return results

# --- 6. Main Execution Flow (核心改动在这里) ---
if __name__ == "__main__":
    print(f"Loading dataset from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"\nOriginal number of rows: {len(df)}")

    # --- 步骤 6A: 预处理和分流 "group" 数据 ---
    # 1. 按类别初步筛选
    df_pre_filter = df.copy()
    print(f"Rows after filtering by category: {len(df_pre_filter)}")
    
    # 2. 定义正则表达式并创建布尔掩码
    group_pattern = r'\b(group of|groups of|bunch of|several|many|herd of|flock of|crowd of|a lot of)\b'
    group_mask = df_pre_filter['caption'].str.contains(group_pattern, case=False, na=False, regex=True)
    
    # 3. 分流出含有 "group" 描述的数据
    df_groups_filtered = df_pre_filter[group_mask]
    
    # 4. 创建最终用于处理的数据集
    df_to_process = df_pre_filter[~group_mask]
    print(f"Found and separated {len(df_groups_filtered)} items with group descriptions.")
    print(f"Remaining items to process: {len(df_to_process)}")

    # --- 步骤 6B: 批量处理和二次分流 ---
    captions_to_process = df_to_process.to_dict('records')
    successful_items = []
    failed_image_ids = []
    
    print(f"\nStarting rewriting process with batch size {BATCH_SIZE}...")
    start_time = time.time()
    
    for i in tqdm(range(0, len(captions_to_process), BATCH_SIZE), desc="Processing Batches"):
        batch_records = captions_to_process[i:i + BATCH_SIZE]
        batch_captions = [rec['caption'] for rec in batch_records]
        
        rewritten_batch = rewrite_batch(batch_captions)
        
        for record, rewritten_text in zip(batch_records, rewritten_batch):
            if rewritten_text.startswith("a photo of"):
                successful_items.append({
                    "file_name": f"{record['image_id']}.jpg", # 假设是jpg格式
                    "text": rewritten_text
                })
            else:
                # 只保存失败条目的 image_id
                failed_image_ids.append(record['image_id'])
        
    end_time = time.time()
    print(f"\n--- Rewriting Finished in {end_time - start_time:.2f} seconds ---")
    print(f"Total items processed: {len(captions_to_process)}")
    print(f"  >> Successful rewrites: {len(successful_items)}")
    print(f"  >> Failed rewrites: {len(failed_image_ids)}")

    # --- 步骤 6C: 保存所有分流文件 ---
    # 1. 保存成功的结果到 JSONL
    with open(SUCCESS_JSONL_PATH, 'w') as f:
        for item in successful_items:
            f.write(f'{{"file_name": "{item["file_name"]}", "text": "{item["text"]}"}}\n')
    print(f"\nSuccessfully saved {len(successful_items)} items to {SUCCESS_JSONL_PATH}")

    # 2. 保存被过滤的 "group" 数据到 CSV
    if not df_groups_filtered.empty:
        df_groups_filtered.to_csv(GROUP_CSV_PATH, index=False)
        print(f"Saved {len(df_groups_filtered)} filtered group-description items to {GROUP_CSV_PATH}")

    # 3. 保存失败的 image_id 到 TXT
    if failed_image_ids:
        with open(FAILED_TXT_PATH, 'w') as f:
            for image_id in failed_image_ids:
                f.write(f"{image_id}\n")
        print(f"Saved {len(failed_image_ids)} failed rewrite IDs to {FAILED_TXT_PATH}")
    else:
        # 仅在没有失败项时打印此消息
        if len(captions_to_process) > 0:
            print("Congratulations! There were no failed rewrites during processing.")