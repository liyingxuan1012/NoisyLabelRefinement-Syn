import os

data_path = 'ImageNet100_noisy/train'

total_images = 0
mismatched_images = 0

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.JPEG'):
                total_images += 1
                if not filename.startswith(folder):
                    mismatched_images += 1

noise_rate = mismatched_images / total_images

print(f'Noise_rate: {mismatched_images} / {total_images} = {noise_rate:.4f}')
