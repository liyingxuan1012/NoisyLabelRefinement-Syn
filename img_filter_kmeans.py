import os
import shutil
import torch
import numpy as np
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def load_images(folder, device):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = preprocess_image(img_path, device)
        images.append(img)
        filenames.append(filename)
    return images, filenames

def extract_features(images, feature_extractor):
    feature_list = []
    with torch.no_grad():
        for img in images:
            feature_map = feature_extractor(img)
            feature_map = feature_map.squeeze(0).cpu().detach().numpy().flatten()
            feature_list.append(feature_map)
    return feature_list

def kmeans_iter0(src_dir, dst_dir, model_path, device):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)

    os.makedirs(dst_dir, exist_ok=True)

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        os.makedirs(dst_path, exist_ok=True)
        print('*' * 50)
        print(class_id, len(os.listdir(src_path)), len(os.listdir(dst_path)))

        if len(os.listdir(src_path))-400 != len(os.listdir(dst_path)):
            # load images and extract features
            real_images, filenames = load_images(src_path, device)
            features = extract_features(real_images, feature_extractor)

            # perform clustering
            kmeans = KMeans(n_clusters=1, random_state=0, n_init=10).fit(features)
            cluster_centers = kmeans.cluster_centers_
            
            # find the images farthest from their cluster centers
            _, distances = pairwise_distances_argmin_min(features, cluster_centers)
            farthest_images = np.argsort(distances)[-400:]

            # remove the images farthest from cluster centers
            removed_images = [filenames[idx] for idx in farthest_images]
            for image in removed_images:
                print(f"Removing {image}: Distance {distances[filenames.index(image)]:.4f}")
            
            # copy the remaining images to the new directory
            for image in filenames:
                if image not in removed_images:
                    src_img_path = os.path.join(src_path, image)
                    dst_img_path = os.path.join(dst_path, image)
                    shutil.copy2(src_img_path, dst_img_path)
        else:
            continue

def kmeans_iter1(src_dir, dst_dir, i_iter, model_path, device):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        print('*' * 50)
        print(class_id, len(os.listdir(src_path))-100*i_iter, len(os.listdir(dst_path)))

        if len(os.listdir(src_path))-100*(i_iter+1) != len(os.listdir(dst_path)):
            # load images and extract features
            real_images, filenames = load_images(src_path, device)
            features = extract_features(real_images, feature_extractor)

            # perform clustering
            kmeans = KMeans(n_clusters=1, random_state=0, n_init=10).fit(features)
            cluster_centers = kmeans.cluster_centers_
            
            # find the images farthest from their cluster centers
            _, distances = pairwise_distances_argmin_min(features, cluster_centers)
            farthest_images = np.argsort(distances)[-400:]

            # remove the images farthest from cluster centers
            removed_images = [filenames[idx] for idx in farthest_images]
            for image in removed_images:
                print(f"Removing {image}: Distance {distances[filenames.index(image)]:.4f}")
                os.remove(os.path.join(dst_path, image))      
        else:
            continue


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models_pretrained/noisy_sym.pt'
    src_dir = '/home/ImageNet100_noisy_sym'
    dst_dir = '/home/ImageNet100_noisy_sym_cluster'
    kmeans_iter0(src_dir, dst_dir, model_path, device)
