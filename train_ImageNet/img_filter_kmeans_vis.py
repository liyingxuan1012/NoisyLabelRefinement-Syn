import os
import torch
import numpy as np
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_images(folder, device):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = preprocess_image(img_path, device)
        images.append(img)
        filenames.append(filename)
    return images, filenames

def extract_features(images, feature_extractor):
    feature_list = []
    with torch.no_grad():
        for img in images:
            feature_map = feature_extractor(img)
            feature_map = feature_map.squeeze(0).cpu().detach().numpy().flatten()
            feature_list.append(feature_map)
    return feature_list


# load fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('models_pretrained/c2.pt', device)
feature_extractor = ResNet50FeatureExtractor(model).to(device)

src_dir = '/scratch/ace14550vm/ImageNet100_noisy/train'
class_id = 'n01440764'
src_path = os.path.join(src_dir, class_id)

# load images and extract features
real_images, filenames = load_images(src_path, device)
features = extract_features(real_images, feature_extractor)

# optional: Reduce dimensionality if features are high-dimensional
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(features)

# perform clustering
kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
labels = kmeans.fit_predict(features_reduced)
cluster_centers = kmeans.cluster_centers_

unique_labels, counts = np.unique(labels, return_counts=True)

# plotting
plt.figure(figsize=(15, 10))
scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
for i, center in enumerate(cluster_centers):
    plt.scatter(center[0], center[1], color='red', s=100, marker='X', label=f'Cluster {i} (n={counts[i]})')
plt.colorbar(scatter, ticks=range(len(unique_labels)))
plt.legend()
plt.savefig('cluster.png')
