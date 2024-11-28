import os
import shutil
import torch
import numpy as np
from PIL import Image
import clip


def preprocess_image(img_path, preprocess_clip):
    image = Image.open(img_path)
    image = preprocess_clip(image).unsqueeze(0).to(device)
    return image

def load_images(folder, preprocess_clip, is_generated=False):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if is_generated or (not is_generated and '_' in filename):
            img_path = os.path.join(folder, filename)
            img = preprocess_image(img_path, preprocess_clip)
            images.append(img)
            filenames.append(filename)
    return images, filenames

def count_real_imgs(directory):
    return len([file for file in os.listdir(directory) if '_' in file])

def compute_average_feature_map(images, clip_model):
    with torch.no_grad():
        image_input = torch.cat(images, dim=0)
        features = clip_model.encode_image(image_input)
        avg_feature_map = features.mean(dim=0)
    return avg_feature_map

def filter_images_iter0(src_dir, dst_dir, device, add_generated=False, discard_count=100, similarity_threshold=0.8):
    # load the CLIP model and preprocessing
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    os.makedirs(dst_dir, exist_ok=True)

    # step 1: extract features for each generated image from all classes
    generated_features = {}
    for class_id in os.listdir(src_dir):
        src_gen_path = f'/home/sd-finetune/data_generated/PMD70_top50_drop-SD-resized/{class_id}'        
        generated_images, _ = load_images(src_gen_path, preprocess_clip, is_generated=True)
        avg_feature_map = compute_average_feature_map(generated_images, clip_model)
        generated_features[class_id] = avg_feature_map.cpu().numpy()

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        os.makedirs(dst_path, exist_ok=True)

        num_real_imgs_src = len(os.listdir(src_path))
        num_real_imgs_dst = count_real_imgs(dst_path)
        print('*' * 50)
        print(f"Processing class '{class_id}': Source images = {num_real_imgs_src}, Destination images = {num_real_imgs_dst}")

        if num_real_imgs_src - discard_count != num_real_imgs_dst:
            # step 2: load real images
            real_images, filenames = load_images(src_path, preprocess_clip, is_generated=False)

            image_labels_confidences = []
            with torch.no_grad():
                for real_img, filename in zip(real_images, filenames):
                    real_img = real_img.to(device)
                    real_feature = clip_model.encode_image(real_img)
                    real_feature = real_feature.squeeze(0).cpu().numpy()

                    # step 3: compute cosine similarity with generated features from all classes
                    similarities = []
                    for gen_class, gen_feature in generated_features.items():
                        similarity = np.dot(real_feature, gen_feature) / (
                            np.linalg.norm(real_feature) * np.linalg.norm(gen_feature))
                        similarities.append(similarity)
                    
                    similarities = np.array(similarities)
                    
                    max_similarity = np.max(similarities)
                    predicted_label = list(generated_features.keys())[np.argmax(similarities)]
                    final_label = predicted_label if max_similarity >= similarity_threshold else class_id
                    image_labels_confidences.append((filename, predicted_label, max_similarity, final_label))

            # step 4: sort by new labels and confidences
            incorrect_images = [item for item in image_labels_confidences if item[3] != class_id]
            correct_images = [item for item in image_labels_confidences if item[3] == class_id]
            # sort correct images by confidence (ascending)
            correct_images.sort(key=lambda x: x[2])

            # step 5: select images to remove
            if len(incorrect_images) >= discard_count:
                images_to_remove = incorrect_images[:discard_count]
            else:
                images_to_remove = incorrect_images + correct_images[:discard_count - len(incorrect_images)]

            for image, predicted_label, confidence, final_label in images_to_remove:
                print(f"Removing {image}: Predicted Label = {predicted_label}, Confidence = {confidence:.4f}, Final Label = {final_label}")

            # copy remaining images to destination directory
            images_to_remove_set = set(item[0] for item in images_to_remove)
            images_to_keep = [fname for fname in filenames if fname not in images_to_remove_set]
            for image in images_to_keep:
                src_img_path = os.path.join(src_path, image)
                dst_img_path = os.path.join(dst_path, image)
                shutil.copy2(src_img_path, dst_img_path)
        else:
            continue

def relabel_and_copy_images(src_dir, dst_dir, device, similarity_threshold=0.8):
    # load the CLIP model and preprocessing
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    os.makedirs(dst_dir, exist_ok=True)

    # step 1: extract features for each generated image from all classes
    generated_features = {}
    for class_id in os.listdir(src_dir):
        src_gen_path = f'/home/sd-finetune/data_generated/PMD70_top50_drop-SD-resized/{class_id}'
        generated_images, _ = load_images(src_gen_path, preprocess_clip, is_generated=True)
        avg_feature_map = compute_average_feature_map(generated_images, clip_model)
        generated_features[class_id] = avg_feature_map.cpu().detach().numpy()

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        os.makedirs(dst_path, exist_ok=True)

        real_images, filenames = load_images(src_path, preprocess_clip, is_generated=False)
        
        # compute similarity to generated features of each class
        image_labels_confidences = []
        with torch.no_grad():
            for real_img, filename in zip(real_images, filenames):
                real_feature = clip_model.encode_image(real_img)
                real_feature = real_feature.squeeze(0).cpu().detach().numpy()
                similarities = []
                for gen_class, gen_feature in generated_features.items():
                    similarity = np.dot(real_feature, gen_feature) / (
                        np.linalg.norm(real_feature) * np.linalg.norm(gen_feature))
                    similarities.append(similarity)

                similarities = np.array(similarities)

                max_similarity = np.max(similarities)
                predicted_label = list(generated_features.keys())[np.argmax(similarities)]
                final_label = predicted_label if max_similarity >= similarity_threshold else class_id
                image_labels_confidences.append((filename, predicted_label, max_similarity, final_label))

        # copy images to new folders based on new labels
        for filename, predicted_label, confidence, final_label in image_labels_confidences:
            dst_class_path = os.path.join(dst_dir, final_label)
            os.makedirs(dst_class_path, exist_ok=True)
            src_img_path = os.path.join(src_path, filename)
            dst_img_path = os.path.join(dst_class_path, filename)
            shutil.copy2(src_img_path, dst_img_path)
            print(f"Copied {filename} to {final_label}: Predicted Label {predicted_label}, Confidence {confidence:.4f}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_dir = '/home/feature-extractor/train_CIFAR/data/CIFAR100_noisy/noisy_PMD70'
    dst_dir = '/home/feature-extractor/train_CIFAR/data/noisy_PMD70_test'
    filter_images_iter0(src_dir, dst_dir, device, add_generated=False, discard_count=300, similarity_threshold=0.8)
    # relabel_and_copy_images(src_dir, dst_dir, device, similarity_threshold=0.8)
