import _init_paths
import os
import os.path as osp
import shutil
import random

import torch
import torchvision
from fire import Fire
from tqdm import tqdm

from utils.utils import  compute_mean_std, get_files

def split_train_val(root, train_percent=0.8):
    train_dir = osp.join(root, '..', 'train')
    val_dir = osp.join(root, '..', 'val')
    class_paths = [osp.join(root, c) for c in os.listdir(root)]

    mkdir_if_not_exist = lambda x: os.makedirs(x) if not osp.isdir(x) else  None
    random.seed(123)
    tq = tqdm(class_paths, total=len(class_paths), desc='Spliting Data')
    for cls_path in tq:
        class_name = os.path.basename(cls_path)
        tq.set_postfix(class_name=class_name)
        train_new_dir = osp.join(train_dir, class_name)
        val_new_dir = osp.join(val_dir, class_name)

        mkdir_if_not_exist(train_new_dir)
        mkdir_if_not_exist(val_new_dir)

        img_files = get_files(cls_path, extensions='.jpg')

        train_class_files = random.sample(
            img_files, int(len(img_files) * train_percent))

        for img_file in img_files:
            if img_file in train_class_files:
                shutil.copy(img_file, osp.join(train_new_dir, os.path.basename(img_file)))
            else:
                shutil.copy(img_file, osp.join(val_new_dir, os.path.basename(img_file)))
        # tq.close()
    return train_dir, val_dir


def main(root, train_percent, bs=128):
    """Create train and val dataset for Mars dataset. Then, compute mean and std
    on train dataset folder.
    Args:
        root - original train folder path
        train_percent - split percent for training, other is validation
        bs - batch size for dataloading
    """
    train_dir, _ = split_train_val(root, train_percent)
    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs)
    mean, std = compute_mean_std(train_loader)
    print('\n'+'-'*50)
    print('Mean: {}'.format(mean))
    print('Std: {}'.format(std))
    print('-'*50)

if __name__ == '__main__':
    Fire(main)
