import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ResNet50FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x

# Load fine-tuned model
model = torch.load('models_tmp/ImageNet100_model_best.pt')
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


real_images_dir = '/home/ace14550vm/ImageNet100/val/n01514668'
generated_images_dir = '/home/ace14550vm/ImageNet100-SD/n01514668'
noise_image_dir = '/home/ace14550vm/ImageNet100/val/n01560419'
noise_generated_image_dir = '/home/ace14550vm/ImageNet100-SD/n01560419'

real_image_paths = get_image_paths(real_images_dir)
generated_image_paths = get_image_paths(generated_images_dir)
noise_image_paths = get_image_paths(noise_image_dir)
noise_generated_image_paths = get_image_paths(noise_generated_image_dir)

# extrac feature
real_features = extract_features(real_image_paths)
generated_features = extract_features(generated_image_paths)
noise_features = extract_features(noise_image_paths)
noise_generated_features = extract_features(noise_generated_image_paths)

all_features = np.vstack([real_features, generated_features, noise_features, noise_generated_features])

# t-SNE
tsne = TSNE(n_components=2, perplexity=50, random_state=0)
reduced_features = tsne.fit_transform(all_features)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[:len(real_features), 0], reduced_features[:len(real_features), 1], c='blue', label='Real Images')
plt.scatter(reduced_features[len(real_features):len(real_features)+len(generated_features), 0], reduced_features[len(real_features):len(real_features)+len(generated_features), 1], c='red', label='Generated Images')
plt.scatter(reduced_features[len(real_features)+len(generated_features):len(real_features)+len(generated_features)+len(noise_features), 0], reduced_features[len(real_features)+len(generated_features):len(real_features)+len(generated_features)+len(noise_features), 1], c='green', label='Noisy Real Images')
plt.scatter(reduced_features[len(real_features)+len(generated_features)+len(noise_features):, 0], reduced_features[len(real_features)+len(generated_features)+len(noise_features):, 1], c='purple', label='Noisy Generated Images')
plt.title('t-SNE Visualization of Image Features')
plt.legend()
plt.savefig('./t_sne_visualization.png')