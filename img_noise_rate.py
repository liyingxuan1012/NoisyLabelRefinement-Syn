import os


def compute_noise_rate(data_path):
    category_stats = {}

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            if folder not in category_stats:
                category_stats[folder] = {'total_images': 0, 'mismatched_images': 0}
            
            for filename in os.listdir(folder_path):
                if '_' in filename:
                    category_stats[folder]['total_images'] += 1
                    if not filename.startswith(folder):
                        category_stats[folder]['mismatched_images'] += 1

    total_images = 0
    mismatched_images = 0

    # calculate and print the noise rate for each category
    for class_id, stats in category_stats.items():
        total_images_cat = stats['total_images']
        mismatched_images_cat = stats['mismatched_images']
        noise_rate_cat = mismatched_images_cat / total_images_cat
        print(f'Class: {class_id}, Noise_rate: {mismatched_images_cat}/{total_images_cat} = {noise_rate_cat:.4f}')

        total_images += total_images_cat
        mismatched_images += mismatched_images_cat
    
    return mismatched_images, total_images


if __name__ == '__main__':
    data_path = 'train_CIFAR/data/CIFAR10_noisy/noisy_PMD35_U60'
    mismatched_images, total_images = compute_noise_rate(data_path)
    
    # calculate and print the overall noise rate
    total_noise_rate = mismatched_images / total_images
    print(f'Total_noise_rate: {mismatched_images} / {total_images} = {total_noise_rate:.4f}')
