import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import numpy as np
import os
import json
import random


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ResNet50FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x

# Load fine-tuned model
model = torch.load('models/ImageNet100_model_best.pt')
feature_extractor = ResNet50FeatureExtractor(model)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        image_paths.append(os.path.join(directory, filename))
    return image_paths

def extract_features(image_paths):
    features = []
    for path in image_paths:
        img = Image.open(path)
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to('cuda')
        with torch.no_grad():
            feature = feature_extractor(img_tensor)
        features.append(feature.squeeze().cpu().numpy())
    return features

def generate_colors(num_classes):
    cmap = mpl.colormaps['tab20']
    colors = cmap(np.linspace(0, 1, num_classes))
    return colors


# load categories
with open('ImageNet100/Labels.json', 'r') as file:
    labels = json.load(file)

# randomly select some categories
num_classes = 10
selected_classes = random.sample(list(labels.keys()), num_classes)
for i, class_id in enumerate(selected_classes):
    class_name = labels[class_id]
    print(f"Class {i+1}, Class ID: {class_id}, Class Name: {class_name}")


# extract features
all_features = []
all_labels = []
for i, class_id in enumerate(selected_classes):
    real_images_dir = f'ImageNet100/val/{class_id}'
    generated_images_dir = f'ImageNet100-SD/{class_id}'

    real_image_paths = get_image_paths(real_images_dir)
    generated_image_paths = get_image_paths(generated_images_dir)

    real_features = extract_features(real_image_paths)
    generated_features = extract_features(generated_image_paths)

    all_features.append(real_features)
    all_features.append(generated_features)

    all_labels.append(f'Class{i+1}_Real')
    all_labels.append(f'Class{i+1}_Generated')


# t-SNE
feature_counts = [len(f) for f in all_features]
all_features = np.vstack(all_features)
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
reduced_features = tsne.fit_transform(all_features)

# visualization
plt.figure(figsize=(15, 12))
colors = generate_colors(num_classes * 2)
current_idx = 0
for i, label in enumerate(all_labels):
    num_features = feature_counts[i]
    end_idx = current_idx + num_features
    plt.scatter(reduced_features[current_idx:end_idx, 0], reduced_features[current_idx:end_idx, 1], color=colors[i], label=label)
    current_idx = end_idx

plt.title('t-SNE Visualization of Image Features')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')
plt.tight_layout()
plt.savefig('./t_sne.png')
