from datasets import load_dataset
import sys

# --- 请在这里修改您的数据集名称或路径 ---
# 如果是 Hugging Face Hub 上的数据集，直接写名称，例如 "lambdalabs/pokemon-blip-captions"
# 如果是本地文件夹，请提供文件夹的路径，例如 "/path/to/your/dataset_folder"
dataset_identifier = "/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin" 
# ------------------------------------

try:
    # 加载数据集
    # 如果您的数据集有多个配置 (config)，您可能需要指定，例如:
    # dataset = load_dataset(dataset_identifier, name="your_config_name")
    dataset = load_dataset(dataset_identifier)

    # 打印数据集的整体结构（会显示有哪些分割，如 'train', 'test'）
    print("\n--- 数据集结构 ---")
    print(dataset)

    # 获取 'train' 分割（通常训练都用这个）
    train_split = dataset['train']

    # 打印 'train' 分割的所有列名
    print("\n--- 'train' 分割的列名 ---")
    print(train_split.column_names)

    # 打印第一条数据，直观地查看每一列的内容
    print("\n--- 数据集第一条示例 ---")
    print(train_split[0])

except Exception as e:
    print(f"\n加载或检查数据集时出错: {e}")
    print("请确认您的数据集名称或路径是否正确，以及网络连接是否正常（如果是在线数据集）。")