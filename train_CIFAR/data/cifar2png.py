import pickle, os
import numpy as np
from PIL import Image


def parse_pickle(rawdata, noisy_labels, rootdir):
    if not os.path.exists(rootdir):
        os.mkdir(rootdir) 
    for i in range(100):
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


with open("cifar-100-python/train", "rb") as fp:
    train = pickle.load(fp, encoding="latin-1")
with open("cifar-100-python/test", "rb") as fp:
    test = pickle.load(fp, encoding="latin-1")

noisy_labels = np.load('cifar100-1-0.70.npy')

parse_pickle(train, noisy_labels, "CIFAR100_noisy/noisy_PMD70")
# parse_pickle(test, "CIFAR100/test")
