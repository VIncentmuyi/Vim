import os
import numpy as np
from PIL import Image

# 定义标签对应的RGB值和类别编号
label_mapping = {
    (0, 0, 0): 0,  # Background clutter
    (128, 0, 0): 1,  # Building
    (128, 64, 128): 2,  # Road
    (0, 128, 0): 3,  # Tree
    (128, 128, 0): 4,  # Low vegetation
    (64, 0, 128): 5,  # Moving car
    (192, 0, 192): 6,  # Static car
    (64, 64, 0): 7  # Human
}


def convert_labels(input_folder):
    # 遍历输入文件夹中的所有PNG文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # 读取原始标签图像
            input_path = os.path.join(input_folder, filename)
            label_image = Image.open(input_path)
            label_array = np.array(label_image)

            # 创建与原图像相同大小的新数组，用于存储类别编号
            label_indices = np.zeros(label_array.shape[:2], dtype=np.uint8)

            # 遍历每个唯一的RGB值，并转换为对应的类别编号
            for rgb, index in label_mapping.items():
                # 找到匹配特定RGB值的像素
                mask = np.all(label_array == rgb, axis=-1)
                label_indices[mask] = index

            # 直接覆盖原文件
            Image.fromarray(label_indices).save(input_path)

            print(f"转换完成: {filename}")


# 使用示例
train_folder = '/root/autodl-tmp/code/vit-adapter/segmentation/data/uavid/annotations/training'
val_folder = '/root/autodl-tmp/code/vit-adapter/segmentation/data/uavid/annotations/validation'

# 转换训练集标签
convert_labels(train_folder)

# 转换验证集标签
convert_labels(val_folder)

print("所有标签转换完成！")