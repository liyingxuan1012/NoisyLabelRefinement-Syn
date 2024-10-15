import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil


# CIFAR-100
transform = transforms.Compose([
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                         std=[0.2675, 0.2565, 0.2761])
])

# load data
data_directory = 'train_CIFAR/data/CIFAR100_noisy/noisy_PMD70'
model_directory = 'train_CIFAR/models_pretrained/cifar100_PMD70.pt'
save_directory = 'PMD70_top50'

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = torch.load(model_directory, map_location=device)
model.eval()

if isinstance(model, nn.DataParallel):
    model = model.module

def predict_and_save_top50(test_directory, save_directory):
    data = datasets.ImageFolder(root=test_directory, transform=transform)
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    class_to_idx = data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # initialize the list of image scores for each class
    class_scores = {idx: [] for idx in class_to_idx.values()}
    test_data = DataLoader(data)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_data)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            scores = torch.max(outputs, 1)[0].cpu().numpy()

            for img_path, label, score in zip(data.imgs[i * test_data.batch_size:(i + 1) * test_data.batch_size], labels, scores):
                class_scores[label.item()].append((img_path[0], score))

    # select the top 50 images with the highest scores and save them
    for class_idx, img_scores in class_scores.items():
        top_50_images = sorted(img_scores, key=lambda x: x[1], reverse=True)[:50]
        class_name = idx_to_class[class_idx]
        class_save_path = os.path.join(save_directory, class_name)
        if not os.path.exists(class_save_path):
            os.makedirs(class_save_path)
        
        print('*' * 50)
        print(f"Top 50 scores for class '{class_name}':")
        for img_path, _ in top_50_images:
            img_name = os.path.basename(img_path)
            save_path = os.path.join(class_save_path, img_name)
            shutil.copy(img_path, save_path)
            print(f"Image: {img_name}, Score: {score:.4f}")

# main
predict_and_save_top50(data_directory, save_directory)
print("Top 50 images per class have been saved.")
