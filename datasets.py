import cv2
import numpy as np
import os
import torch

from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.filenames = []
        self.labels = []
        for i, cat in enumerate(os.listdir(path)):
            cat_path = os.path.join(path, cat)
            lbs = [i] * len(os.listdir(cat_path))
            self.labels += lbs
            fns = os.listdir(cat_path)
            fns = [os.path.join(cat_path, fn) for fn in fns]
            self.filenames += fns

        print(f"Found {len(self.filenames)} images belonging to {len(np.unique(np.array(self.labels)))} classes.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        x = cv2.imread(fn, cv2.IMREAD_COLOR)
        try:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(fn)
        if self.transform:
            x = self.transform(x)

        y = torch.tensor(self.labels[idx])
        return x, y
