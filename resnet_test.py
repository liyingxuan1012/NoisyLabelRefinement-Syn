import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
 
 
def pridict():
    # image preprocessing
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = 'ImageNet100'
    # test_directory = os.path.join(dataset, 'val')
    test_directory = os.path.join('ImageNet100-SD')
    model_directory = os.path.join('models_tmp', dataset + '_model_best.pt')

    data = datasets.ImageFolder(root=test_directory, transform=transform)
    data_size = len(data)
    test_data = DataLoader(data)

    # class_to_idx = data.class_to_idx
    # idx_to_class = dict((val, key) for key, val in class_to_idx.items())
    # print(idx_to_class)
    
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = torch.load(model_directory)
    model = model.to(device)
 
    model.eval()


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
