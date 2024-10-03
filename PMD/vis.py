import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# 定义图像预处理，去除归一化以便于图片显示
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# 加载 CIFAR100 数据集
train_dataset = datasets.CIFAR100(root='data', train=True, transform=transform, download=False)

# 计算前90%的索引
split_index = int(0.9 * len(train_dataset))

# 加载含噪声的标签
noise_labels = np.load('data/cifar100-1-0.35.npy')

# 找出原始标签和噪声标签不一致的图片索引
mismatched_indices = [idx for idx, label in enumerate(train_dataset.targets[:split_index]) if label != noise_labels[idx]]

# 从第一个类别的图片中随机选择10个索引
selected_indices = random.sample(mismatched_indices, 16)

# 准备展示图片和标签
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 5))  # 调整网格大小和图片尺寸
axes = axes.flatten()  # 将 axes 数组展平，便于索引

for i, ax in enumerate(axes):
    idx = selected_indices[i]
    img, _ = train_dataset[idx]
    img = img.permute(1, 2, 0)  # 调整通道顺序以适配 matplotlib

    original_label = train_dataset.classes[train_dataset.targets[idx]]  # 获取原始标签名
    noisy_label = train_dataset.classes[noise_labels[idx]]  # 获取噪声标签名

    ax.imshow(img)
    ax.axis('off')  # 关闭坐标轴
    ax.set_title(f"{original_label} / {noisy_label}")

plt.tight_layout()

# 保存图片到文件
plt.savefig('data/mismatched_labels_comparison.png')  # 修改路径以匹配您的保存位置

# 清理图形对象，防止重复使用内存
plt.close()
