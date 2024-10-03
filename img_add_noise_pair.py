import os
import random
import shutil


# original dataset
dataset_path = 'data/CIFAR100/train'
categories = os.listdir(dataset_path)

# create noisy dataset
noisy_dataset_path = 'data/CIFAR100_noisy/noisy_pair60'
os.makedirs(noisy_dataset_path, exist_ok=True)
noise_rate = 0.6

# assign each category to the next category in the shuffled list as noise
random.shuffle(categories)
noisy_target_categories = categories[1:] + categories[:1]
noise_mapping = dict(zip(categories, noisy_target_categories))
# print(noise_mapping)

for category in categories:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)
    
    # calculate the number of images to add noise to
    num_noisy_images = int(len(images) * noise_rate)
    noisy_images = random.sample(images, num_noisy_images)
    
    # map noisy images to the specific target category
    noise_image_allocation = {image: noise_mapping[category] for image in noisy_images}

    # copy images to new directory and assign noisy labels to selected images
    for image in images:
        src_path = os.path.join(category_path, image)
        new_category = noise_image_allocation.get(image, category)

        new_category_path = os.path.join(noisy_dataset_path, new_category)
        os.makedirs(new_category_path, exist_ok=True)
        
        dst_path = os.path.join(new_category_path, image)
        shutil.copyfile(src_path, dst_path)
