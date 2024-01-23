import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# image preprocessing
transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

# load data
dataset = 'ImageNet100'
real_directory = os.path.join(dataset, 'val')
generated_directory = os.path.join('ImageNet100-SD')
model_directory = os.path.join('models', dataset + '_model_best.pt')

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = torch.load(model_directory, map_location=device)
model.eval()


def pridict(test_directory):
    data = datasets.ImageFolder(root=test_directory, transform=transform)
    class_to_idx = data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    data_size = len(data)
    test_data = DataLoader(data)
    
    # initialize class accuracy tracking
    class_correct = {idx: [0, 0] for idx in class_to_idx.values()}
    total_correct = 0

    # test
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            for label, correct in zip(labels, correct_counts):
                class_correct[label.item()][0] += correct.item()
                class_correct[label.item()][1] += 1
                total_correct += correct.item()

    # calculate and print class accuracies
    class_accuracies = []
    class_ids = []
    for idx, (correct, total) in class_correct.items():
        accuracy = correct / total
        class_accuracies.append(accuracy)
        class_ids.append(idx_to_class[idx])
        # print(f"Class '{idx_to_class[idx]}' Accuracy: {accuracy:.4f}")
        
    # calculate and print total accuracy
    avg_test_acc = total_correct / data_size
    print("Total Accuracy: {} / {} = {:.4f}".format(int(total_correct), data_size, avg_test_acc))

    return class_ids, class_accuracies


def plot_class_accuracies(real_class_ids, real_class_acc, generated_class_ids, generated_class_acc):
    assert real_class_ids == generated_class_ids
    num_classes = len(real_class_ids)

    width = 0.3
    index = np.arange(num_classes)
    
    plt.figure(figsize=(num_classes * 0.5, 5))
    plt.bar(index - width/2, real_class_acc, width, label='Real Images')
    plt.bar(index + width/2, generated_class_acc, width, label='Generated Images')

    plt.xlabel('Class ID')
    plt.ylabel('Accuracy')
    plt.title('Class Accuracies')
    plt.xticks(index, real_class_ids, rotation=60)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')
    plt.xlim(min(index) - 1, max(index + width) + 1)


    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    plt.close()


# main
real_class_ids, real_class_acc = pridict(real_directory)
generated_class_ids, generated_class_acc = pridict(generated_directory)
plot_class_accuracies(real_class_ids, real_class_acc, generated_class_ids, generated_class_acc)
