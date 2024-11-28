import os
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import time
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
args = parser.parse_args()

# # CIFAR-10
# image_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize(size=32),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
# }

# CIFAR-100
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                             std=[0.2675, 0.2565, 0.2761])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                             std=[0.2675, 0.2565, 0.2761])
    ])
}

# configure logging
logging.basicConfig(filename=args.log_dir, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def train(epochs):
    # load data
    num_gpus = torch.cuda.device_count()
    train_data = DataLoader(data['train'], batch_size=args.batch_size, shuffle=True, num_workers=4*num_gpus)
    valid_data = DataLoader(data['valid'], batch_size=args.batch_size, shuffle=False, num_workers=4*num_gpus)
    test_data = DataLoader(data['test'], batch_size=args.batch_size, shuffle=False, num_workers=4*num_gpus)

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model_directory = args.model_dir
    if os.path.exists(model_directory):
        model = torch.load(model_directory, map_location=device)
        logging.info("Load model from {}".format(model_directory))
        fc_inputs = model.fc[0].in_features
    else:
        from resnet import resnet34
        model = resnet34()
        # model = models.resnet34(weights=None)
        logging.info("Initialize ResNet-34 from scratch")
        fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, args.num_classes)

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    iter_per_epoch = len(train_data)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    history = []
    best_acc = 0.0
    best_epoch = 0

    # train
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            
            if epoch <= args.warm:
                warmup_scheduler.step()

        if epoch > args.warm:
            scheduler.step()
        
        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model, model_directory)
            # test
            test_acc = test(model, test_data, device)

        epoch_end = time.time()

        logging.info("Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        logging.info("Best Accuracy for validation: {:.4f} at epoch {:03d}, Test Accuracy: {:.4f}".format(best_acc, best_epoch, test_acc))

    return model, history

def test(model, test_data, device):
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            test_acc += torch.sum(correct_counts.type(torch.FloatTensor))

    test_acc = test_acc / len(test_data.dataset)
    return test_acc

def plot_curve(history):
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Train Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig(args.train_dir + '_loss_curve.png')
    plt.close()

    plt.plot(history[:, 2:4])
    plt.legend(['Train Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.savefig(args.train_dir + '_accuracy_curve.png')
    plt.close()


# load datasets
data = {
    'train': datasets.ImageFolder(root=args.train_dir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root="data/CIFAR100/val", transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root="data/CIFAR100/test", transform=image_transforms['valid'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

num_epochs = 200
trained_model, history = train(num_epochs)
plot_curve(history)
