import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
 
 
def pridict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = './models/data_model_best.pt'
 
    model = torch.load(path)
    model = model.to(device)
 
    model.eval()
 
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5125,0.4667,0.4110],
                             std=[0.2621,0.2501,0.2453])
    ])

    dataset = './data'
    test_directory = os.path.join(dataset, 'test')
    data = datasets.ImageFolder(root=test_directory, transform=transform)
    data_size = len(data)

    # class_to_idx = data.class_to_idx
    # idx_to_class = dict((val, key) for key, val in class_to_idx.items())
    # print(idx_to_class)
    
    test_data = DataLoader(data)
    test_acc = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            ret, predictions = torch.max(outputs.data, 1)
            # print(name, labels.cpu().numpy(), predictions.cpu().numpy())
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            test_acc += acc.item() * inputs.size(0)
    avg_test_acc = test_acc / data_size
    print("Accuracy: {} / {} = {:.4f}".format(int(test_acc), data_size, avg_test_acc))

 
if __name__ == '__main__':
    pridict()
