# -*- coding: utf-8 -*-
from PIL import Image
from torch.utils.data import DataLoader

if __package__:
    from .basedataset import BaseDataset
else:
    import sys
    sys.path.insert(0, '..')
    import torch
    import torchvision
    from torchvision import transforms
    from basedataset import BaseDataset

class Market1501(BaseDataset):
    def __getitem__(self, index):
        img_fs, lbl = self.data.imgs[index]
        img = Image.open(img_fs).convert('RGB')
        img = self.tfms(img) if self.tfms else img
        return img, lbl


if __name__ == '__main__':
    root = '/home/zmh/hdd/Custom_Projects/object_tracking/datasets/Market_1501'
    H, W = 256, 128
    tfms = transforms.Compose([
        transforms.Resize((H, W)), # h,w
        transforms.ToTensor()
        ])
    train_dataset = Market1501(root, mode='train', tfms=tfms)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    imgs, lbls = next(iter(train_loader))
    if isinstance(imgs, (tuple, list)):
        imgs = torch.cat([imgs[0], imgs[1], imgs[2]], dim=-1)
    grid = torchvision.utils.make_grid(imgs, nrow=imgs.shape[0]//6)
