import os
import sys
import random
import shutil
import torch
import numpy as np

sys.path.append('../')
from feature_extractor import ResNet50FeatureExtractor, load_model, preprocess_image


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

def filter_images_iter0(src_dir, dst_dir, model_path, device, add_generated=False, discard_count=100, similarity_threshold=0.8):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)
    model.eval()

    os.makedirs(dst_dir, exist_ok=True)

    # step 1: extract features for each generated image from all classes
    generated_features = {}
    for class_id in os.listdir(src_dir):
        src_gen_path = f'/home/SD-xl-turbo/train/{class_id}'
        generated_images, _ = load_images(src_gen_path, device, is_generated=True)
        generated_features[class_id] = compute_average_feature_map(generated_images, feature_extractor).cpu().detach().numpy().flatten()

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        os.makedirs(dst_path, exist_ok=True)

        num_real_imgs_src = len(os.listdir(src_path))
        num_real_imgs_dst = count_real_imgs(dst_path)
        print('*' * 50)
        print(class_id, num_real_imgs_src, num_real_imgs_dst)

        if num_real_imgs_src - discard_count != num_real_imgs_dst:
            # step 2: load real images and extract features
            real_images, filenames = load_images(src_path, device, is_generated=False)
            
            image_labels_confidences = []
            for real_img, filename in zip(real_images, filenames):
                real_feature = feature_extractor(real_img)
                real_feature = real_feature.squeeze(0).cpu().detach().numpy().flatten()
    
                # step 3: compute cosine similarity with generated features from all classes
                similarities = []
                for gen_class, gen_feature in generated_features.items():
                    similarity = np.dot(real_feature, gen_feature) / (np.linalg.norm(real_feature) * np.linalg.norm(gen_feature))
                    similarities.append(similarity)
                
                similarities = np.array(similarities)
                
                # step 4: predict label using classifier scores + similarity score
                with torch.no_grad():
                    outputs = model(real_img)
                    classifier_scores = torch.softmax(outputs, dim=1).cpu().detach().numpy().flatten()
                
                # ensure scores are summed for the same class
                final_confidences = []
                for i, gen_class in enumerate(generated_features.keys()):
                    final_confidence = 0.5 * similarities[i] + 0.5 * classifier_scores[int(gen_class)]
                    final_confidences.append(final_confidence)
                
                final_confidences = np.array(final_confidences)
                max_confidence = np.max(final_confidences)
                predicted_label = list(generated_features.keys())[np.argmax(final_confidences)]
                final_label = predicted_label if max_confidence >= similarity_threshold else class_id
                image_labels_confidences.append((filename, predicted_label, max_confidence, final_label))

            # step 5: sort by new labels and confidences
            incorrect_images = [item for item in image_labels_confidences if item[3] != class_id]
            correct_images = [item for item in image_labels_confidences if item[3] == class_id]
            # sort correct images by confidence (ascending)
            correct_images.sort(key=lambda x: x[2])

            # step 6: select images to remove
            images_to_remove = incorrect_images[:discard_count] if len(incorrect_images) >= discard_count else incorrect_images + correct_images[:discard_count - len(incorrect_images)]
            for image, predicted_label, confidence, final_label in images_to_remove:
                print(f"Removing {image}: Predicted Label {predicted_label}, Final Confidence {confidence:.4f}, Final Label {final_label}")

            # remove selected images and copy remaining images to destination directory
            images_to_keep = set(filenames) - set([item[0] for item in images_to_remove])
            for image in images_to_keep:
                src_img_path = os.path.join(src_path, image)
                dst_img_path = os.path.join(dst_path, image)
                shutil.copy2(src_img_path, dst_img_path)

            # add generated images if specified
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

def relabel_and_copy_images(src_dir, dst_dir, model_path, device, similarity_threshold=0.6):
    model = load_model(model_path, device)
    feature_extractor = ResNet50FeatureExtractor(model).to(device)
    model.eval()

    os.makedirs(dst_dir, exist_ok=True)

    # step 1: extract features for each generated image from all classes
    generated_features = {}
    for class_id in os.listdir(src_dir):
        src_gen_path = f'/home/SD-xl-turbo/train/{class_id}'
        generated_images, _ = load_images(src_gen_path, device, is_generated=True)
        generated_features[class_id] = compute_average_feature_map(generated_images, feature_extractor).cpu().detach().numpy().flatten()

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        os.makedirs(dst_path, exist_ok=True)

        num_real_imgs_src = len(os.listdir(src_path))
        num_real_imgs_dst = count_real_imgs(dst_path)
        print('*' * 50)
        print(class_id, num_real_imgs_src, num_real_imgs_dst)

        # step 2: load real images and extract features
        real_images, filenames = load_images(src_path, device, is_generated=False)

        image_labels_confidences = []
        for real_img, filename in zip(real_images, filenames):
            real_feature = feature_extractor(real_img)
            real_feature = real_feature.squeeze(0).cpu().detach().numpy().flatten()

            # step 3: compute cosine similarity with generated features from all classes
            similarities = []
            for gen_class, gen_feature in generated_features.items():
                similarity = np.dot(real_feature, gen_feature) / (np.linalg.norm(real_feature) * np.linalg.norm(gen_feature))
                similarities.append(similarity)
            
            similarities = np.array(similarities)
            
            # step 4: Predict label using classifier scores + similarity score
            with torch.no_grad():
                outputs = model(real_img)
                classifier_scores = torch.softmax(outputs, dim=1).cpu().detach().numpy().flatten()
            
            # ensure scores are summed for the same class
            final_confidences = []
            for i, gen_class in enumerate(generated_features.keys()):
                final_confidence = 0.5 * similarities[i] + 0.5 * classifier_scores[int(gen_class)]
                final_confidences.append(final_confidence)
            
            final_confidences = np.array(final_confidences)
            max_confidence = np.max(final_confidences)
            predicted_label = list(generated_features.keys())[np.argmax(final_confidences)]
            final_label = predicted_label if max_confidence >= similarity_threshold else class_id
            image_labels_confidences.append((filename, predicted_label, max_confidence, final_label))

        # step 5: copy images to new folders based on new labels
        for filename, predicted_label, confidence, final_label in image_labels_confidences:
            dst_class_path = os.path.join(dst_dir, final_label)
            os.makedirs(dst_class_path, exist_ok=True)
            src_img_path = os.path.join(src_path, filename)
            dst_img_path = os.path.join(dst_class_path, filename)
            shutil.copy2(src_img_path, dst_img_path)
            print(f"Copied {filename} to {final_label}: Predicted Label {predicted_label}, Confidence {confidence:.4f}")

def random_discard_images(src_dir, dst_dir, discard_count=100):
    os.makedirs(dst_dir, exist_ok=True)

    for class_id in os.listdir(src_dir):
        src_path = os.path.join(src_dir, class_id)
        dst_path = os.path.join(dst_dir, class_id)
        os.makedirs(dst_path, exist_ok=True)

        image_files = os.listdir(src_path)
        images_to_keep = random.sample(image_files, len(image_files)-discard_count)

        for image in images_to_keep:
            src_img_path = os.path.join(src_path, image)
            dst_img_path = os.path.join(dst_path, image)
            shutil.copy2(src_img_path, dst_img_path)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/feature-extractor/train_ImageNet/models_pretrained/noisy_pair60.pt'
    src_dir = '/home/feature-extractor/train_ImageNet/data/ImageNet100_noisy/noisy_pair60'
    dst_dir = '/home/feature-extractor/train_ImageNet/data/imagenet100_pair60'
    # filter_images_iter0(src_dir, dst_dir, model_path, device, add_generated=False, discard_count=100, similarity_threshold=0.8)
    relabel_and_copy_images(src_dir, dst_dir, model_path, device, similarity_threshold=0)
