import torch
import matplotlib.pyplot as plt
import numpy as np
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image


# visualize features
def plot_feature_maps(layers_output, selected_channels, real_image):
    num_layers = len(layers_output)
    num_columns = len(selected_channels) + 1
    fig, axes = plt.subplots(num_layers, num_columns, figsize=(num_columns * 2, num_layers * 2))
    fig.suptitle('Feature Maps per Layer', fontsize=16)

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
        
        # show other columns using selected_channels
        for j, channel in enumerate(selected_channels):
            if channel < feature_maps.shape[0]:
                channel_feature_map_normalized = normalize_feature_map(feature_maps[channel])
                ax = axes[i, j + 1]
                ax.imshow(channel_feature_map_normalized, cmap='viridis')
                ax.axis('off')
                if i == 0:
                    ax.set_title(f"Channel {channel}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if real_image:
        plt.savefig('feature_maps_real.png')
    else:
        plt.savefig('feature_maps_generated.png')
    plt.close()

def normalize_feature_map(feature_map):
    return (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-5)

# find top N channels with the highest values in the first layer
def find_top_channels(layers_output, n_channels):
    first_layer = layers_output[0].squeeze(0).cpu().detach().numpy()
    summed_feature_map = np.sum(first_layer, axis=(1, 2))
    top_channels = np.argsort(summed_feature_map)[-n_channels:][::-1]
    return top_channels


# load fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('models/ImageNet100_model_best.pt', device)
feature_extractor = ResNet50FeatureExtractor(model, extract_layers=True).to(device)

# process real image
real_img_path = 'ImageNet100/val/n01531178/ILSVRC2012_val_00003548.JPEG'
real_image = preprocess_image(real_img_path, device)
with torch.no_grad():
    real_layers_output = feature_extractor(real_image)

# process generated image
generated_img_path = 'ImageNet100-SD/n01531178/00037.png'
generated_image = preprocess_image(generated_img_path, device)
with torch.no_grad():
    generated_layers_output = feature_extractor(generated_image)

# select channels
num_channels = 10
selected_channels = find_top_channels(real_layers_output, num_channels)

# visualize feature maps
plot_feature_maps(real_layers_output, selected_channels, real_image=True)
plot_feature_maps(generated_layers_output, selected_channels, real_image=False)
