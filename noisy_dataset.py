import os
import random
import shutil


# original dataset
dataset_path = 'ImageNet100/train'
categories = os.listdir(dataset_path)

# create noisy dataset
noisy_dataset_path = 'ImageNet100_noisy/train'
os.makedirs(noisy_dataset_path, exist_ok=True)
noise_rate = 0.3

for category in categories:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    
    # calculate the number of images to add noise to
    num_noisy_images = int(len(images) * noise_rate)
    noisy_images = random.sample(images, num_noisy_images)
    
    # copy images to new directory and assign noisy labels to selected images
    for image in images:
        source_path = os.path.join(category_path, image)
        if image in noisy_images:
            new_category = category
            while new_category == category:
                new_category = random.choice(categories)
        else:
            new_category = category
        
        new_category_path = os.path.join(noisy_dataset_path, new_category)
        os.makedirs(new_category_path, exist_ok=True)
        
        destination_path = os.path.join(new_category_path, image)
        shutil.copyfile(source_path, destination_path)
