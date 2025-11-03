import os
from datasets import load_dataset

# 定义输出目录
output_dir = "/data/home/scyb226/lzx/data/megatron"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 下载 codeparrot-clean-train 数据集
train_data = load_dataset('codeparrot/codeparrot-clean-train', split='train')

# 保存为 JSONL 格式，每行一个样本
output_file = os.path.join(output_dir, "codeparrot_data.json")
train_data.to_json(output_file, lines=True)
print(f"数据已保存到: {output_file}")
