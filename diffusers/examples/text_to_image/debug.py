import datasets
from PIL import Image
import os

# --- 配置 ---
# 请将此路径指向您想查看的任何Arrow数据集目录
DATASET_DIR = '/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin/data/train/' # 或者是 'final_dataset_for_training' 等

# 您想查看第几条数据？(0代表第一条)
INDEX_TO_VIEW = 0

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at '{DATASET_DIR}'.")
        exit()

    print(f"--- Loading dataset from '{DATASET_DIR}' ---")
    dataset = datasets.load_from_disk(DATASET_DIR)
    
    print("\n--- Dataset Information ---")
    print(dataset)
    
    if INDEX_TO_VIEW >= len(dataset):
        print(f"\nError: Index {INDEX_TO_VIEW} is out of bounds. The dataset has {len(dataset)} rows.")
        exit()
        
    print(f"\n--- Displaying content of row #{INDEX_TO_VIEW} ---")
    
    # 获取指定行的数据
    single_row = dataset[INDEX_TO_VIEW]
    
    # 打印所有列和它们的值
    for column_name, value in single_row.items():
        print(f"  Column '{column_name}':")
        # 对不同类型的值进行友好展示
        if isinstance(value, str):
            # 如果是字符串，打印前200个字符
            print(f"    Type: String")
            print(f"    Value: '{value[:200]}{'...' if len(value) > 200 else ''}'")
        elif isinstance(value, Image.Image):
            # 如果是Pillow对象，打印它的基本信息
            print(f"    Type: PIL.Image.Image")
            print(f"    Mode: {value.mode}, Size: {value.size}")
        else:
            # 其他类型直接打印
            print(f"    Type: {type(value).__name__}")
            print(f"    Value: {value}")
            
    # 特别检查 'image' 列
    print("\n--- Detailed check of 'image' column ---")
    image_column_value = single_row['image']
    print(f"The Python type of the 'image' column is: {type(image_column_value)}")
    if isinstance(image_column_value, str):
        print("It's a string, likely a file path.")
    elif isinstance(image_column_value, Image.Image):
        print("It's a Pillow Image object, which is what the training script likely expects.")