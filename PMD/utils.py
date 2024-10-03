from torch.utils.data import Dataset
from PIL import Image


class Custom_dataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.n = len(list(targets))
        self.index = list(range(self.n))
        self.transform = transform

    def __getitem__(self, i):
        img = self.data[i]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[i], self.index[i]

    def __len__(self):
        return self.n

    def update_label(self, noise_label):
        self.targets[:] = noise_label[:]
