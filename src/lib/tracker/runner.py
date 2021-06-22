# -*- coding: utf-8 -*-
import os
import time
from itertools import product
from collections import namedtuple, OrderedDict

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = [Run(*v) for v in product(*params.values())]
        return runs


class Runner:
    def __init__(self, train_meta, data_meta, verbose=True):
        self.train_meta = train_meta
        self.data_meta = data_meta
        self.verbose = verbose
        self.save_root = self.create_save_root(train_meta.save_root)

        self.tb = None
        self.epoch_count = 0
        self.total_loss = 0.
        self.total_accuracy = 0

        self.run_params = None
        self.run_start_time = None
        self.run_count = 0

    def create_save_root(self, path):
        root = os.path.join(*path) \
            if isinstance(path, (list, tuple)) else path

        get_last = lambda x: sorted(os.listdir(x), key=lambda f: int(f.split('-')[-1]))[-1]
        run_id = int(get_last(root).split('-')[-1]) if os.path.isdir(root) else 0
        save_root = os.path.join(root, f'runs-{run_id+1}')
        os.makedirs(save_root)
        if self.verbose: print(f'[INFO] Creating training checkpoints folder : {save_root}')
        return save_root

    def begin_run(self, run):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        log_dir = os.path.join(self.save_root, 'logs', f'{self.run_count}-{run}')
        self.tb = SummaryWriter(log_dir=log_dir)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        self.run_start_time = None

    def begin_dataiter(self):
        self.epoch_count += 1
        self.total_loss = 0
        self.total_accuracy = 0

    def end_dataiter(self, loader, model, epoch, prefix_tag):
        self.total_accuracy /= len(loader.dataset)
        self.total_loss /= len(loader.dataset)
        imgs, lbls = next(iter(loader))
        if isinstance(imgs, (tuple, list)): # triplet dataloader
            self.tb.add_graph(model.cpu(), [imgs[0][0:1], imgs[1][0:1], imgs[2][0:1]]) #TODO cuda
            imgs = torch.cat([imgs[0], imgs[1], imgs[2]], dim=-1)
        else:
            self.tb.add_graph(model.cpu(), imgs[0:1])
        grid = torchvision.utils.make_grid(imgs, nrow=imgs.shape[0]//6)
        self.tb.add_image(f'{prefix_tag} images', grid, epoch)             #
        self.tb.add_scalar(f'{prefix_tag} loss', self.total_loss, epoch)
        self.tb.add_scalar(f'{prefix_tag} Acc', self.total_accuracy, epoch)
        # reset tracked variables
        self.total_accuracy = 0
        self.total_loss = 0

    def track_loss(self, loss, images):
        self.total_loss += loss.item() * images.shape[0]

    def track_metric(self, preds, labels):
        self.total_accuracy += preds.argmax(dim=1).eq(labels).sum().item()


if __name__ == '__main__':
    params = OrderedDict(
        lr = [0.1, 0.01],
        loss = ['entropy', 'dice']
        )
    runs = RunBuilder.get_runs({params})