# -*- coding: utf-8 -*-
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

if __package__:
    from .utils import get_transforms
    from .basedataset import BaseDataset
    from .utils import show_tensor
else:
    from utils import get_transforms
    from basedataset import BaseDataset
    from utils import *


class Mars(BaseDataset):
    def __getitem__(self, index):
        img_fs, lbl = self.data.imgs[index]
        img = Image.open(img_fs).convert('RGB')
        img = self.tfms(img) if self.tfms else img
        return img, lbl


if __name__ == '__main__':
    root = '/home/zmh/hdd/Custom_Projects/object_tracking/datasets/Mars'
    H, W = 256, 128
    tfms = get_transforms(H, W)
    train_dataset = Mars(root, mode='train', tfms=tfms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    img, lbl = next(iter(train_loader))
    grid = make_grid(img, nrow=8)
    show_tensor(grid)
