from transformers import AutoModel, AutoTokenizer

# 指定下载目录
model_path = "./models/Qwen3-8B"

model = AutoModel.from_pretrained(
    "Qwen/Qwen3-8B",
    cache_dir=model_path  # 指定下载位置
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-8B", 
    cache_dir=model_path  # 指定下载位置
)