import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random


# 定义图像预处理，去除归一化以便于图片显示
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# 加载 CIFAR10 数据集
train_dataset = datasets.CIFAR100(root='.', train=True, transform=transform, download=False)
# 加载含噪声的标签
noise_labels = np.load('data/noise_label/cifar10-1-0.35.npy')

# 计算前90%的索引
split_index = int(0.9 * len(train_dataset))
# 找出原始标签和噪声标签不一致的图片索引
mismatched_indices = [idx for idx, label in enumerate(train_dataset.targets[:split_index]) if label != noise_labels[idx]]

# 从第一个类别的图片中随机选择10个索引
selected_indices = random.sample(mismatched_indices, 16)

# 准备展示图片和标签
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 5))
axes = axes.flatten()

for i, ax in enumerate(axes):
    idx = selected_indices[i]
    img, _ = train_dataset[idx]
    img = img.permute(1, 2, 0)

    original_label = train_dataset.classes[train_dataset.targets[idx]]
    noisy_label = train_dataset.classes[noise_labels[idx]]

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"{original_label} / {noisy_label}")

plt.tight_layout()

# 保存图片到文件
plt.savefig('mismatched_labels_comparison.png')
plt.close()
