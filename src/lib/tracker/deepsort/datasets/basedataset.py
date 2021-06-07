# -*- coding: utf-8 -*-
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root, mode, tfms=None):
        self.mode = mode
        self.root = os.path.join(root, mode)
        self.tfms = tfms
        self.data = ImageFolder(self.root)
        self.num_classes = len(self.data.classes)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def normalize(self, mean, std):
        self.norm = transforms.Normalize(mean, std)