import os
import sys
import shutil
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

sys.path.append('../')
from feature_extractor import ResNet50FeatureExtractor, load_model


def preprocess_image(img_path, device):
    transform = transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

def load_images(folder, device, is_generated=False):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if is_generated or (not is_generated and '_' in filename):
            img_path = os.path.join(folder, filename)
            img = preprocess_image(img_path, device)
            images.append(img)
            filenames.append(filename)
    return images, filenames

def count_real_imgs(directory):
    return len([file for file in os.listdir(directory) if '_' in file])

def compute_similarities(real_images, filenames, generated_images, feature_extractor):
    # compute average feature maps for generated images
    avg_generated_feature_map = compute_average_feature_map(generated_images, feature_extractor)
    avg_generated_feature_map = avg_generated_feature_map.squeeze(0).cpu().detach().numpy().flatten()
    
    # compute cosine similarity for each real image
    image_similarities = []
    for real_img, filename in zip(real_images, filenames):
        real_feature_map = feature_extractor(real_img)
        real_feature_map = real_feature_map.squeeze(0).cpu().detach().numpy().flatten()

        cos_sim = np.dot(real_feature_map, avg_generated_feature_map) / (np.linalg.norm(real_feature_map) * np.linalg.norm(avg_generated_feature_map))
        image_similarities.append((filename, cos_sim))
    return image_similarities

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

def add_generated_images(src_gen_dir, dst_dir, iter_idx):
    start_index = 100 * iter_idx
    end_index = start_index + 100
    for i in range(start_index, end_index):
        src_img_path = os.path.join(src_gen_dir, f'{i:05d}.png')
        dst_img_path = os.path.join(dst_dir, f'{i:05d}.png')
        shutil.copy2(src_img_path, dst_img_path)

def filter_images_iter0(src_dir, dst_dir, model_path, device, add_generated=False):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)

    os.makedirs(dst_dir, exist_ok=True)

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        src_gen_path = f'/home/CIFAR100-SD/{class_id}'
        os.makedirs(dst_path, exist_ok=True)
        
        num_real_imgs_src = len(os.listdir(src_path))
        num_real_imgs_dst = count_real_imgs(dst_path)
        print('*' * 50)
        print(class_id, num_real_imgs_src, num_real_imgs_dst)

        if num_real_imgs_src-100 != num_real_imgs_dst:
            # load images
            real_images, filenames = load_images(src_path, device, is_generated=False)
            generated_images, _ = load_images(src_gen_path, device, is_generated=True)

            # compute cosine similarity for each real image against the average generated feature map
            image_similarities = compute_similarities(real_images, filenames, generated_images, feature_extractor)
            image_similarities.sort(key=lambda x: x[1])
            lowest_similarity_images = image_similarities[:100]
            for image, sim in lowest_similarity_images:
                print(f"Removing {image}: Similarity {sim:.4f}")

            # remove the images with the lowest cosine similarity
            top_images = image_similarities[100:]
            for image, _ in top_images:
                src_img_path = os.path.join(src_path, image)
                dst_img_path = os.path.join(dst_path, image)
                shutil.copy2(src_img_path, dst_img_path)
            
            # add generated images
            if add_generated:
                add_generated_images(src_gen_path, dst_path, 0)
        else:
            continue

def filter_images_iter1(src_dir, dst_dir, i_iter, model_path, device, add_generated=False):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        src_gen_path = f'/home/CIFAR100-SD/{class_id}'

        num_real_imgs_src = len(os.listdir(src_path))
        num_real_imgs_dst = count_real_imgs(dst_path)
        print('*' * 50)
        print(class_id, num_real_imgs_src-100*i_iter, num_real_imgs_dst)
        if add_generated:
            assert len(os.listdir(src_path)) == len(os.listdir(dst_path))

        if num_real_imgs_src-100*(i_iter+1) != num_real_imgs_dst:
            # load images
            real_images, filenames = load_images(dst_path, device, is_generated=False)
            generated_images, _ = load_images(src_gen_path, device, is_generated=True)

            # compute cosine similarity for each real image against the average generated feature map
            image_similarities = compute_similarities(real_images, filenames, generated_images, feature_extractor)
            image_similarities.sort(key=lambda x: x[1])
            lowest_similarity_images = image_similarities[:100]
            for image, sim in lowest_similarity_images:
                print(f"Removing {image}: Similarity {sim:.4f}")
                os.remove(os.path.join(dst_path, image))
 
            # add generated images
            if add_generated:
                add_generated_images(src_gen_path, dst_path, i_iter)
        else:
            continue


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models_pretrained/noisy_sym60.pt'
    src_dir = '/home/ImageNet100_noisy/noisy_sym60'
    dst_dir = '/home/ImageNet100_noisy_sym60_onestep'
    filter_images_iter0(src_dir, dst_dir, model_path, device, add_generated=False)
