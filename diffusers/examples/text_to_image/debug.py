import os
import csv
from datasets import load_dataset
from PIL import Image
import sys

# --- 1. 配置区域 ---

# 数据集路径 (已根据您的提供进行设置)
dataset_path = "/inspire/hdd/project/25jinqiu14/public/datasets_new/vismin"

# 您想要处理并保存的样本数量
# 如果想处理整个数据集，请注意可能需要很长时间和大量磁盘空间
num_samples_to_save = 50 

# 输出文件的文件夹名称
output_dir = "dataset_preview"

# --- 2. 主逻辑区域 ---

# 创建输出文件夹（如果它还不存在）
os.makedirs(output_dir, exist_ok=True)
print(f"✅ 输出文件夹 '{output_dir}' 已准备就绪。")

# --- 加载真实数据集 ---
print(f"⏳ 正在尝试从 '{dataset_path}' 加载数据集...")
# 加载 'train' 分割
dataset = load_dataset(dataset_path, split="train")
print(f"✅ 数据集加载成功！共找到 {len(dataset)} 条数据。")

# --- 准备写入元数据文件 ---
metadata_file_path = os.path.join(output_dir, "metadata.csv")

# 检查数据集是否为空
if not dataset:
    print("❌ 数据集为空，无法继续。")
    sys.exit()

# 打开CSV文件准备写入
with open(metadata_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    # 从第一条数据中动态获取所有非图片内容的列名作为表头
    first_item_keys = dataset[0].keys()
    header = [key for key in first_item_keys if key != 'image']
    
    # 创建CSV写入器
    writer = csv.DictWriter(csvfile, fieldnames=header)
    # 写入表头
    writer.writeheader()
    
    print(f"⏳ 开始处理并保存前 {num_samples_to_save} 个样本...")
    
    # --- 遍历数据集并保存 ---
    # 使用 enumerate 和切片来只处理指定数量的样本
    for i, item in enumerate(dataset.select(range(num_samples_to_save))):
        
        # 从数据项中获取图片对象和ID
        image_obj = item.get('image')
        image_id = item.get('image_id') # 使用 image_id 作为文件名
        
        # 安全检查
        if not isinstance(image_obj, Image.Image) or not image_id:
            print(f"⚠️ 跳过第 {i+1} 条记录，因为它缺少有效的 'image' 或 'image_id'。")
            continue
            
        # 根据图片格式确定文件扩展名
        file_extension = ".png" if image_obj.format == 'PNG' else ".jpg"
        image_filename = f"{image_id}{file_extension}"
        image_save_path = os.path.join(output_dir, image_filename)
        
        # 1. 保存图片文件
        image_obj.save(image_save_path)
        
        # 2. 准备元数据并写入CSV
        metadata_to_write = {key: value for key, value in item.items() if key != 'image'}
        writer.writerow(metadata_to_write)
        
        # 打印进度
        print(f"  -> 已保存图片: {image_filename}")

print("\n🎉 全部处理完成！")
print(f"🖼️  {num_samples_to_save} 张图片已保存到 '{output_dir}' 文件夹。")
print(f"📄 对应的元数据已保存到 '{metadata_file_path}' 文件中。")