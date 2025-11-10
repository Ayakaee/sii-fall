import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import time
import re
import datasets
import os

# --- 1. Configuration ---
MODEL_PATH = "Qwen/Qwen3-8B"  
INPUT_DATASET_DIR = '/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin/data/train' # 您的原始Arrow数据集目录

# --- 输出文件路径配置 ---
OUTPUT_DATASET_DIR = '/inspire/hdd/project/25jinqiu14/sunyihang-P-253130146/dataset/' 
SUCCESS_JSONL_PATH = 'intermediate_successful.jsonl' # 【中间文件】增量写入的成功结果
GROUP_CSV_PATH = 'filtered_groups_log.csv'
FAILED_TXT_PATH = 'failed_rewrites_ids.txt'
PROGRESS_LOG_PATH = 'processed_ids.log' # 【新】进度跟踪文件
BATCH_SIZE = 256

# --- Helper function to load processed IDs ---
def load_processed_ids(log_path):
    if not os.path.exists(log_path):
        return set()
    with open(log_path, 'r') as f:
        # 使用set以便快速查找
        return set(line.strip() for line in f)

# --- 2. Load Model & Tokenizer (不变) ---
# ... (这部分代码完全不变)
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
# ... (这部分代码完全不变)
class SuppressThinkProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.forbidden_token_id = tokenizer.encode("<|think|>", add_special_tokens=False)[0]
    def __call__(self, input_ids, scores):
        scores[:, self.forbidden_token_id] = -float("inf")
        return scores
logits_processor_list = LogitsProcessorList([SuppressThinkProcessor(tokenizer)])

# --- 4. Meta-Prompt (不变) ---
# ... (这部分代码完全不变)
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
# ... (这部分代码完全不变)
def rewrite_batch(captions):
    prompts = [create_qwen_prompt(cap) for cap in captions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=False, logits_processor=logits_processor_list
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
    # --- 步骤 6A: 加载并进行初步过滤 ---
    print(f"Loading dataset from disk: {INPUT_DATASET_DIR}...")
    raw_dataset = datasets.load_from_disk(INPUT_DATASET_DIR)
    df_full = raw_dataset.to_pandas()
    print(f"\nFull dataset loaded with {len(df_full)} rows.")

    # --- 步骤 6B: 处理恢复逻辑 ---
    processed_ids = load_processed_ids(PROGRESS_LOG_PATH)
    print(f"Found {len(processed_ids)} already processed IDs. Skipping them.")
    
    # 过滤掉所有已处理的ID
    df_remaining = df_full[~df_full['image_id'].isin(processed_ids)].copy()

    # if df_remaining.empty:
    #     print("All items have already been processed.")
    # else:
    #     print(f"Items remaining to process: {len(df_remaining)}")
    
    #     # --- 步骤 6C: 对剩余数据进行"group"过滤 ---
    #     group_pattern = r'\b(group of|groups of|bunch of|several|many|herd of|flock of|crowd of|a lot of)\b'
    #     group_mask = df_remaining['caption'].fillna('').str.contains(group_pattern, case=False, regex=True)
        
    #     df_groups_to_log = df_remaining[group_mask]
    #     df_to_process = df_remaining[~group_mask]
        
    #     if not df_groups_to_log.empty:
    #         # 以追加模式写入，避免重复写入
    #         df_groups_to_log.to_csv(GROUP_CSV_PATH, mode='a', header=not os.path.exists(GROUP_CSV_PATH), index=False)
    #         print(f"Logged {len(df_groups_to_log)} new items with group descriptions to {GROUP_CSV_PATH}")
    #         # 将这些ID也加入进度日志
    #         with open(PROGRESS_LOG_PATH, 'a') as f:
    #             for image_id in df_groups_to_log['image_id']:
    #                 f.write(f"{image_id}\n")
        
    #     print(f"Items to rewrite in this run: {len(df_to_process)}")

    #     # --- 步骤 6D: 批量处理和增量保存 ---
    #     if not df_to_process.empty:
    #         captions_to_process = df_to_process.to_dict('records')
            
    #         print(f"\nStarting rewriting process...")
    #         start_time = time.time()
            
    #         for i in tqdm(range(0, len(captions_to_process), BATCH_SIZE), desc="Processing Batches"):
    #             batch_records = captions_to_process[i:i + BATCH_SIZE]
    #             batch_captions = [str(rec.get('caption', '')) for rec in batch_records]
                
    #             rewritten_batch = rewrite_batch(batch_captions)
                
    #             # 准备本批次要写入文件的内容
    #             success_to_append = []
    #             failed_to_append = []
    #             processed_ids_this_batch = []

    #             for record, rewritten_text in zip(batch_records, rewritten_batch):
    #                 image_id = record['image_id']
    #                 processed_ids_this_batch.append(image_id)
    #                 if rewritten_text.startswith("a photo of"):
    #                     success_to_append.append(f'{{"file_name": "{image_id}.jpg", "text": "{rewritten_text}"}}\n')
    #                 else:
    #                     failed_to_append.append(f"{image_id}\n")
                
    #             # --- 原子化写入，确保数据安全 ---
    #             if success_to_append:
    #                 with open(SUCCESS_JSONL_PATH, 'a') as f:
    #                     f.writelines(success_to_append)
    #             if failed_to_append:
    #                 with open(FAILED_TXT_PATH, 'a') as f:
    #                     f.writelines(failed_to_append)
    #             # 最后更新进度日志
    #             with open(PROGRESS_LOG_PATH, 'a') as f:
    #                 for pid in processed_ids_this_batch:
    #                     f.write(f"{pid}\n")
            
    #         end_time = time.time()
    #         print(f"\n--- This run finished in {end_time - start_time:.2f} seconds ---")

    # --- 步骤 6E: 最终组装 (只有当所有工作都完成时才需要) ---
    final_processed_count = len(load_processed_ids(PROGRESS_LOG_PATH))
    if final_processed_count != len(df_full):
        print("\n--- All items processed. Assembling final Arrow dataset. ---")
        if os.path.exists(SUCCESS_JSONL_PATH):
            # 从增量写入的JSONL文件加载数据集
            final_ds = datasets.load_dataset('json', data_files=SUCCESS_JSONL_PATH, split='train')
            
            # 使用map函数高效地替换列
            def remap_columns(example):
                return {
                    'caption': example['text'] # 将text内容赋给caption
                }
            
            # 移除旧的text列，并重命名
            final_ds = final_ds.map(remap_columns, remove_columns=['text'])
            
            # 如果原始数据集中除了'caption'和'image_id'还有其他列，需要在这里决定是否保留
            # 例如，只保留这两列：
            # final_ds = final_ds.select_columns(['image_id', 'caption'])
            
            if os.path.exists(OUTPUT_DATASET_DIR):
                print(f"Warning: Output directory '{OUTPUT_DATASET_DIR}' already exists. It will be overwritten.")
            final_ds.save_to_disk(OUTPUT_DATASET_DIR)
            print(f"Successfully saved final Arrow dataset with {len(final_ds)} items to '{OUTPUT_DATASET_DIR}'")
            print("Schema of the new dataset:")
            print(final_ds.info)
        else:
            print("No successful items found to assemble.")
    else:
        print(f"\nProcess paused. {final_processed_count}/{len(df_full)} items processed. Run the script again to continue.")