import pickle, os
import numpy as np


def parse_images_to_npy(rootdir, rawdata, output_path):
    # Create an array to store new noisy labels
    new_noisy_labels = np.zeros(len(rawdata["filenames"]), dtype=int)
    
    # Loop through the images in original order
    m = len(rawdata["filenames"])
    for i in range(m-5000):
        filename = rawdata["filenames"][i]
        gt_label = rawdata["fine_labels"][i]
        
        # Search for the image in the root directory
        found = False
        for class_dir in os.listdir(rootdir):
            img_path = os.path.join(rootdir, class_dir, f"{gt_label:02d}_{filename}")
            if os.path.exists(img_path):
                new_label = int(class_dir)
                new_noisy_labels[i] = new_label
                found = True
                break
        
        if not found:
            raise FileNotFoundError(f"File not found for: {gt_label:02d}_{filename}")
    
    # Save the new labels as .npy file
    np.save(output_path, new_noisy_labels)

def calculate_noise_rate(labels_path, gt_labels):
    labels = np.load(labels_path)
    m = len(gt_labels)
    mismatches = np.sum(labels[:m-5000] != gt_labels[:m-5000])
    noise_rate = mismatches / (m - 5000)
    return noise_rate


# # CIFAR-100
# with open("train_CIFAR/data/cifar-100-python/train", "rb") as fp:
#     train = pickle.load(fp, encoding="latin-1")

# CIFAR-10
train_data = []
for i in range(1, 6):
    with open(f"train_CIFAR/data/cifar-10-batches-py/data_batch_{i}", "rb") as fp:
        batch = pickle.load(fp, encoding="latin-1")
        train_data.append(batch)

train = {"filenames": [], "fine_labels": []}
for batch in train_data:
    train["filenames"].extend(batch["filenames"])
    train["fine_labels"].extend(batch["labels"])


# Parse images to npy file
parse_images_to_npy("train_CIFAR/data/cifar10_PMD35_A30_0.6", train, "train_CIFAR/data/noise_label/cifar10-1-0.35_A_0.3-ours-0.6.npy")

# Calculate noise rates
gt_labels = np.array(train["fine_labels"])
noise_rate_original = calculate_noise_rate("train_CIFAR/data/noise_label/cifar10-1-0.35_A_0.3.npy", gt_labels)
noise_rate_ours = calculate_noise_rate("train_CIFAR/data/noise_label/cifar10-1-0.35_A_0.3-ours-0.6.npy", gt_labels)

print(f"Noise rate before denoising: {noise_rate_original:.2%}")
print(f"Noise rate after denoising: {noise_rate_ours:.2%}")
