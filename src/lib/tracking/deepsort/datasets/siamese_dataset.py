import os
import time
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

if __package__:
    from .basedataset import BaseDataset
    from .utils import *
else:
    from basedataset import BaseDataset
    from utils import *


class SiameseTriplet(BaseDataset):
    def __getitem__(self, index):
        img0_fs, img0_lbl = self.dataset.imgs[index]
        while True:
            # get negative sample
            img1_fs, img1_lbl = random.choice(self.dataset.imgs)
            if img1_lbl != img0_lbl:
                break

        # get same person files related to anchor image
        this_person_folder = os.path.dirname(img0_fs)
        this_person_fs = [
            os.path.join(this_person_folder, file)
            for file in os.listdir(this_person_folder)
            if file != os.path.basename(img0_fs)
            ]

        # get 1 random same person file via the above person files
        positive_fs = random.choice(this_person_fs) \
            if len(this_person_fs) > 0 else img0_fs

        assert os.path.dirname(img0_fs) == os.path.dirname(positive_fs), 'Error! anchor and postive id not same'
        anchor = Image.open(img0_fs).convert('RGB')
        positive = Image.open(positive_fs).convert('RGB')
        negative = Image.open(img1_fs).convert('RGB')

        if self.tfms:
            anchor, positive, negative = map(self.tfms, [anchor, positive, negative])

        return anchor, positive, negative

if __name__ == '__main__':
    root = '/home/zmh/hdd/Custom_Projects/object_tracking/datasets/Market_1501'
    H, W = 256, 128
    tfms = transforms.Compose([
        transforms.Resize((H, W)), # h,w
        transforms.ToTensor()
        ])

    tic = time.time()
    # we use DatasetFolder class which is used to calculate mean and std
    train_dataset = SiameseTriplet(root, mode='train', tfms=tfms)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=1)
    end = time.time() - tic
    print(f'\nLoading Time : {end:.4f}s')
