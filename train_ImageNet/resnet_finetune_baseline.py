import os
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging
import argparse
from tqdm import tqdm
from img_noise_rate import compute_noise_rate
from img_filter_baseline import baseline_iter0, baseline_iter1
from img_filter_kmeans import kmeans_iter0, kmeans_iter1


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--valid_dir', type=str, required=True)
parser.add_argument('--pretrained_model_dir', type=str, required=True)
parser.add_argument('--best_model_dir', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
args = parser.parse_args()


# image preprocessing
# image_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
# }
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                             std=[0.2675, 0.2565, 0.2761])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                             std=[0.2675, 0.2565, 0.2761])
    ])
}


def train_and_valid(data, model, device, loss_function, optimizer, scheduler, epochs):
    # load data
    num_gpus = torch.cuda.device_count()
    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=4*num_gpus)
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=4*num_gpus)

    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])
    history = []
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

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
        # scheduler.step()

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

        # scheduler step is based on validation accuracy
        scheduler.step(avg_valid_acc)
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"Current Learning Rate: {current_lr}")

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model, best_model_directory)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 25:
                logging.info("Early stopping as no improvement in validation accuracy")
                break
            
        epoch_end = time.time()

        logging.info("Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        logging.info("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    return model, history

def iterative_process(initial_data_dir, pretrained_model_dir, start_iter, num_iter):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    for i in range(start_iter, num_iter):
        logging.info("********** Iteration{} Start **********".format(i+1))
        
        # load model
        model_directory = pretrained_model_dir if i == 0 else best_model_directory
        if os.path.exists(model_directory):
            model = torch.load(model_directory, map_location='cpu')
            logging.info("Load model from {}".format(model_directory))
        else:
            raise FileNotFoundError("Model directory does not exist: {}".format(model_directory))

        original_model = model.module if isinstance(model, nn.DataParallel) else model
        fc_inputs = original_model.fc.in_features
        # original_model.fc = nn.Sequential(
        #     nn.Linear(fc_inputs, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, num_classes),
        #     nn.LogSoftmax(dim=1)
        # )
        original_model.fc = nn.Linear(fc_inputs, num_classes)

        model = original_model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs")

        loss_function = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

        # remove the images with the lowest cosine similarity
        if i == 0:
            # baseline_iter0(initial_data_dir, train_directory, model_directory, device)
            kmeans_iter0(initial_data_dir, train_directory, model_directory, device)
        else:
            # baseline_iter1(initial_data_dir, train_directory, i, model_directory, device)
            kmeans_iter1(initial_data_dir, train_directory, i, model_directory, device)

        # calculate the noise rate of filtered images
        mismatched_images, total_images = compute_noise_rate(train_directory)
        total_noise_rate = mismatched_images / total_images
        logging.info("Noise rate: {} / {} = {:.4f}".format(mismatched_images, total_images, total_noise_rate))
        
        data = {
            'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
            'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
        }

        model, history = train_and_valid(data, model, device, loss_function, optimizer, scheduler, num_epochs)
        logging.info("********** Iteration{} End **********".format(i+1))


if __name__ == '__main__':
    # configure logging
    logging.basicConfig(filename=args.log_dir, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


    batch_size = 128
    num_classes = 100
    num_epochs = 50
    start_iteration = 0
    num_iterations = 3


    dataset_directory = args.dataset_dir
    train_directory = args.train_dir
    valid_directory = args.valid_dir
    pretrained_model_directory = args.pretrained_model_dir
    best_model_directory = args.best_model_dir

    iterative_process(dataset_directory, pretrained_model_directory, start_iteration, num_iterations)
