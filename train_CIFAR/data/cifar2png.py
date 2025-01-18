import pickle, os
import numpy as np
from PIL import Image


def parse_pickle(rawdata, num_class, noisy_labels, rootdir):
    if not os.path.exists(rootdir):
        os.mkdir(rootdir) 
    for i in range(num_class):
        classdir = rootdir + "/" + f"{i:02d}"
        if not os.path.exists(classdir):
            os.mkdir(classdir)    

    m = len(rawdata["filenames"])
    for i in range(m-5000):
        filename = rawdata["filenames"][i]
        gt_label = rawdata["fine_labels"][i]
        if noisy_labels is not None:
            label = noisy_labels[i]
        else:
            label = gt_label

        data = rawdata["data"][i]
        data = data.reshape(3, 32, 32)
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 0, 1)
        with Image.fromarray(data) as img:
            img.save(f"{rootdir}/{label:02d}/{gt_label:02d}_{filename}")


# # CIFAR-100
# with open("train_CIFAR/data/cifar-100-python/train", "rb") as fp:
#     train = pickle.load(fp, encoding="latin-1")
# # with open("train_CIFAR/data/cifar-100-python/test", "rb") as fp:
# #     test = pickle.load(fp, encoding="latin-1")

# noisy_labels = np.load('train_CIFAR/data/noise_label/cifar100-1-0.35_A_0.3.npy')

# parse_pickle(train, 100, noisy_labels, "train_CIFAR/data/CIFAR100_noisy/noisy_PMD35_A30")
# # parse_pickle(test, 100, None, "CIFAR100/test")


# CIFAR-10
train_data = []
for i in range(1, 6):
    with open(f"train_CIFAR/data/cifar-10-batches-py/data_batch_{i}", "rb") as fp:
        batch = pickle.load(fp, encoding="latin-1")
        train_data.append(batch)

merged_train_data = {
    "filenames": [],
    "fine_labels": [],
    "data": np.empty((0, 3072), dtype=np.uint8)
}

for batch in train_data:
    merged_train_data["filenames"].extend(batch["filenames"])
    merged_train_data["fine_labels"].extend(batch["labels"])
    merged_train_data["data"] = np.vstack((merged_train_data["data"], batch["data"]))

noisy_labels = np.load('train_CIFAR/data/noise_label/cifar10-1-0.35_A_0.3.npy')

parse_pickle(merged_train_data, 10, noisy_labels, "train_CIFAR/data/CIFAR10_noisy/noisy_PMD35_A30")
