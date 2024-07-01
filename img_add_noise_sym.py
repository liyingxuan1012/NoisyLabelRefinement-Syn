import os
import random
import shutil


# original dataset
dataset_path = '/home/ImageNet100/train'
categories = os.listdir(dataset_path)

# create noisy dataset
noisy_dataset_path = '/home/ImageNet100_noisy_sym'
os.makedirs(noisy_dataset_path, exist_ok=True)
noise_rate = 0.3

for category in categories:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    
    # calculate the number of images to add noise to
    num_noisy_images = int(len(images) * noise_rate)
    noisy_images = random.sample(images, num_noisy_images)
    
    # calculate the number of noisy images per remaining category
    remaining_categories = [cat for cat in categories if cat != category]
    num_images_per_category = num_noisy_images // len(remaining_categories)
    extra_images = num_noisy_images % len(remaining_categories)

    # distribute images evenly and handle the extras
    noise_allocation = {cat: num_images_per_category for cat in remaining_categories}
    extra_categories = random.sample(remaining_categories, extra_images)
    for cat in extra_categories:
        noise_allocation[cat] += 1

    # map noisy images to new categories
    noise_image_allocation = {}
    for cat, count in noise_allocation.items():
        for _ in range(count):
            if noisy_images:
                selected_image = noisy_images.pop()
                noise_image_allocation[selected_image] = cat

    # copy images to new directory and assign noisy labels to selected images
    for image in images:
        src_path = os.path.join(category_path, image)
        new_category = noise_image_allocation.get(image, category)

        new_category_path = os.path.join(noisy_dataset_path, new_category)
        os.makedirs(new_category_path, exist_ok=True)
        
        dst_path = os.path.join(new_category_path, image)
        shutil.copyfile(src_path, dst_path)
