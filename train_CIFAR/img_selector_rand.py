import os
import random
import shutil


# 原始数据集路径和目标路径
source_dir = 'train_CIFAR/data/CIFAR100_noisy/noisy_PMD70'
target_dir = 'noisy_PMD70_rand50'

# 创建目标路径下的子文件夹
os.makedirs(target_dir, exist_ok=True)

# 遍历每个子文件夹
for class_folder in os.listdir(source_dir):
    source_calss_path = os.path.join(source_dir, class_folder)
    if os.path.isdir(source_calss_path):
        # 获取该子文件夹中的所有图片
        images = os.listdir(source_calss_path)
        # 随机选取50张图片
        selected_images = random.sample(images, 50)
        
        # 创建目标子文件夹
        target_class_path = os.path.join(target_dir, class_folder)
        os.makedirs(target_class_path, exist_ok=True)
        
        # 将选取的图片复制到目标子文件夹
        for image in selected_images:
            src_image_path = os.path.join(source_calss_path, image)
            target_image_path = os.path.join(target_class_path, image)
            shutil.copy(src_image_path, target_image_path)

print("图片选择和复制完成！")
