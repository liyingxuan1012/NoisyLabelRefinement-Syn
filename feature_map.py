import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image


# visualize features
def plot_feature_maps(layers_output, num_columns, channels_per_layer):
    num_layers = len(layers_output)
    fig, axes = plt.subplots(num_layers, num_columns, figsize=(num_columns * 2, num_layers * 2))
    fig.suptitle('Feature Maps per Layer', fontsize=16)

    # select channels
    num_channels_first_layer = layers_output[0].size(1)
    selected_channels = random.sample(range(num_channels_first_layer), min(channels_per_layer, num_channels_first_layer))

    for i, feature_maps in enumerate(layers_output):
        feature_maps = feature_maps.squeeze(0).cpu().detach().numpy()
        summed_feature_map = np.sum(feature_maps, axis=0)
        summed_feature_map_normalized = normalize_feature_map(summed_feature_map)
        
        # show the 1st column
        axes[i, 0].imshow(summed_feature_map_normalized, cmap='viridis')
        axes[i, 0].set_yticks([])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_frame_on(False)
        axes[i, 0].set_ylabel(f"Layer {i+1}")
        if i == 0:
            axes[i, 0].set_title("Sum")
        
        # show other columns
        for j, channel in enumerate(selected_channels):
            if channel < feature_maps.shape[0]:
                channel_feature_map_normalized = normalize_feature_map(feature_maps[channel])
                ax = axes[i, j + 1]
                ax.imshow(channel_feature_map_normalized, cmap='viridis')
                ax.axis('off')
                if i == 0:
                    ax.set_title(f"Channel {channel}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('feature_maps.png')
    plt.close()

# draw a histogram of the summed feature map (layer 8)
def plot_layer_histogram(layers_output):
    num_layers = len(layers_output)
    feature_map = np.sum(layers_output[num_layers-2].squeeze(0).cpu().detach().numpy(), axis=0)
    feature_map_normalized = normalize_feature_map(feature_map)
    flattened_feature_map = feature_map_normalized.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(flattened_feature_map, bins=30, rwidth=0.9)
    plt.title(f"Histogram of Summed Feature Map - Layer 8")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f'layer8_histogram.png')
    plt.close()

def normalize_feature_map(feature_map):
    return (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-5)


# load fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('models/ImageNet100_model_best.pt', device)
feature_map_extractor = ResNet50FeatureExtractor(model, extract_layers=True).to(device)

# load image and extract features
img_path = 'ImageNet100-SD/n01491361/00007.png'
image = preprocess_image(img_path, device)
with torch.no_grad():
    layers_output = feature_map_extractor(image)

# main
plot_feature_maps(layers_output, num_columns=11, channels_per_layer=10)
plot_layer_histogram(layers_output)