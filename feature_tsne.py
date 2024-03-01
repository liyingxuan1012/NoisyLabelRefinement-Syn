import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image


def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        image_paths.append(os.path.join(directory, filename))
    return image_paths

def extract_features(image_paths):
    features = []
    for path in image_paths:
        img_tensor = preprocess_image(path, device)
        feature = feature_extractor(img_tensor)
        features.append(feature.squeeze().cpu().detach().numpy())
    return features

def generate_colors(num_classes):
    cmap = mpl.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, num_classes))
    return colors


# load fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('models/ImageNet100_model_best.pt', device)
feature_extractor = ResNet50FeatureExtractor(model).to(device)

# load categories
with open('ImageNet100/Labels.json', 'r') as file:
    labels = json.load(file)

# select categories
selected_classes = ['n01773549', 'n01775062']
num_classes = len(selected_classes)
for i, class_id in enumerate(selected_classes):
    class_name = labels[class_id]
    print(f"Class ID: {class_id}, Class Name: {class_name}")

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

    all_labels.append(f'{class_id}_Real')
    all_labels.append(f'{class_id}_Generated')

# t-SNE
feature_counts = [len(f) for f in all_features]
all_features = np.vstack(all_features)
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
reduced_features = tsne.fit_transform(all_features)

# visualization
plt.figure(figsize=(10, 8))
colors = generate_colors(num_classes)
current_idx = 0
for i, label in enumerate(all_labels):
    num_features = feature_counts[i]
    end_idx = current_idx + num_features
    # use the same color index for each class
    color_idx = i // 2
    if i % 2 == 0:
        plt.scatter(reduced_features[current_idx:end_idx, 0], reduced_features[current_idx:end_idx, 1], color=colors[color_idx], label=label)
    else:
        plt.scatter(reduced_features[current_idx:end_idx, 0], reduced_features[current_idx:end_idx, 1], color=colors[color_idx], alpha=0.5, label=label)
    current_idx = end_idx

plt.title('t-SNE Visualization of Image Features')
plt.legend(loc='best', fontsize='large')
plt.tight_layout()
plt.savefig('feature_tsne.png')
