import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image


def load_images(folder, device):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = preprocess_image(img_path, device)
        images.append(img)
    return images

def compute_average_feature_map(images, feature_extractor):
    sum_feature_maps = None
    with torch.no_grad():
        for img in images:
            feature_map = feature_extractor(img)
            if sum_feature_maps is None:
                sum_feature_maps = feature_map
            else:
                sum_feature_maps += feature_map
    avg_feature_map = sum_feature_maps / len(images)
    return avg_feature_map

def compute_cos_sim(real_layers_output, generated_layers_output):
    real_feature_maps = real_layers_output.squeeze(0).cpu().detach().numpy()
    real_feature_maps = real_feature_maps.flatten()
    generated_feature_maps = generated_layers_output.squeeze(0).cpu().detach().numpy()
    generated_feature_maps = generated_feature_maps.flatten()

    cos_sim = np.dot(real_feature_maps, generated_feature_maps) / (np.linalg.norm(real_feature_maps) * np.linalg.norm(generated_feature_maps))
    return cos_sim

def plot_layer_histogram(real_layers_output, generated_layers_output, n_channels):
    real_summed_values = []
    generated_summed_values = []

    # extract feature maps
    real_feature_maps = real_layers_output.squeeze(0).cpu().detach().numpy()
    generated_feature_maps = generated_layers_output.squeeze(0).cpu().detach().numpy()

    # find top N channels with the highest values
    summed_feature_map = np.sum(real_feature_maps, axis=(1, 2))
    selected_channels = np.argsort(summed_feature_map)[-n_channels:][::-1]

    for channel in selected_channels:
        if channel < real_feature_maps.shape[0]:
            # sum feature map values for each channel
            real_summed_values.append(np.sum(real_feature_maps[channel]))
            generated_summed_values.append(np.sum(generated_feature_maps[channel]))
    
    # normalize the values
    # real_summed_values = normalize_feature_map(np.array(real_summed_values))
    # generated_summed_values = normalize_feature_map(np.array(generated_summed_values))

    # plotting
    width = 0.3
    x = np.arange(len(selected_channels))

    plt.figure(figsize=(len(selected_channels) * 0.5, 6))
    plt.bar(x - width/2, real_summed_values, width, label='Real  Images (Classifier1)')
    plt.bar(x + width/2, generated_summed_values, width, label='Generated  Images (Classifier1)')

    plt.xlabel('Channel')
    plt.ylabel('Summed Value')
    plt.title('Summed Values per Channel')
    plt.xticks(x, selected_channels)
    plt.legend(fontsize='large')
    plt.xlim(min(x) - 1, max(x + width) + 1)

    plt.tight_layout()
    plt.savefig('feature_histogram.png')
    plt.close()

def normalize_feature_map(feature_map):
    return (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-5)


# load fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('models/ImageNet100_model_best.pt', device)
feature_extractor = ResNet50FeatureExtractor(model).to(device)

# load images
class_id = 'n01775062'
real_images = load_images(f'ImageNet100/val/{class_id}', device)
generated_images = load_images(f'ImageNet100-SD/{class_id}', device)

# compute average feature maps
avg_real_feature_map = compute_average_feature_map(real_images, feature_extractor)
avg_generated_feature_map = compute_average_feature_map(generated_images, feature_extractor)

# plot the histogram of values
num_channels = 30
plot_layer_histogram(avg_real_feature_map, avg_generated_feature_map, num_channels)

# # compute cosine similarity
# cos_sim = compute_cos_sim(avg_real_feature_map, avg_generated_feature_map)
# print(cos_sim)
