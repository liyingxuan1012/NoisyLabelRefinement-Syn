import os
import shutil
import torch
import numpy as np
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image


def load_images(folder, device):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = preprocess_image(img_path, device)
        images.append(img)
        filenames.append(filename)
    return images, filenames

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

def baseline_iter0(src_dir, dst_dir, model_path, device):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)

    os.makedirs(dst_dir, exist_ok=True)

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        os.makedirs(dst_path, exist_ok=True)
        print('*' * 50)
        print(class_id, len(os.listdir(src_path)), len(os.listdir(dst_path)))

        if len(os.listdir(src_path))-100 != len(os.listdir(dst_path)):
            # load images
            real_images, filenames = load_images(src_path, device)
            generated_images = real_images

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
        else:
            continue

def baseline_iter1(src_dir, dst_dir, i_iter, model_path, device):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        print('*' * 50)
        print(class_id, len(os.listdir(src_path))-100*i_iter, len(os.listdir(dst_path)))

        if len(os.listdir(src_path))-100*(i_iter+1) != len(os.listdir(dst_path)):
            # load images
            real_images, filenames = load_images(dst_path, device)
            generated_images = real_images

            # compute cosine similarity for each real image against the average generated feature map
            image_similarities = compute_similarities(real_images, filenames, generated_images, feature_extractor)
            image_similarities.sort(key=lambda x: x[1])
            lowest_similarity_images = image_similarities[:100]
            for image, sim in lowest_similarity_images:
                print(f"Removing {image}: Similarity {sim:.4f}")
                os.remove(os.path.join(dst_path, image))
        else:
            continue


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models_pretrained/noisy_sym.pt'
    src_dir = '/home/test'
    dst_dir = '/home/test_iter1'
    baseline_iter0(src_dir, dst_dir, model_path, device)
