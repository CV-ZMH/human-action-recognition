# -*- coding: utf-8 -*-
import os
import time
import json
from datetime import datetime
from collections import namedtuple, OrderedDict
from itertools import product

import pandas as pd
from tabulate import tabulate
import torchvision
from torch.utils.tensorboard import SummaryWriter


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = [Run(*v) for v in product(*params.values())]
        return runs


class Runner:
    def __init__(self, tb_folder):
        self.epoch_count = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_start_time = None
        self.run_count = 0
        self.run_data = []

        self.network = None
        self.tb = None
        self.tb_folder = tb_folder

        self.train_loader = None
        self.train_duration = None
        self.train_loss = 0
        self.train_accuracy = 0
        self.val_loader = None
        self.val_duration = None
        self.val_loss = 0
        self.val_accuracy = 0
        self.total_num_correct = 0
        self.total_loss = 0

    def begin_run(self, run, network, train_loader, val_loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(self.tb_folder, f'{current_time}-{run}')
        self.tb = SummaryWriter(log_dir=log_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.total_loss = 0
        self.total_num_correct = 0

    def add_tb_data(self, status):
        "Call after training loop or validation loop"
        if status == 'train':
            self.train_loss = self.total_loss / len(self.train_loader.dataset)
            self.train_accuracy = self.total_num_correct / len(self.train_loader.dataset)
            self.tb.add_scalar('Train Acc', self.train_accuracy, self.epoch_count)
            self.tb.add_scalar('Train Loss', self.train_loss, self.epoch_count)
            # only calculate training loop duration
            self.train_duration = time.time() - self.epoch_start_time
            self.epoch_start_time = time.time()

        elif status == 'val':
            self.val_loss = self.total_loss / len(self.val_loader.dataset)
            self.val_accuracy = self.total_num_correct / len(self.val_loader.dataset)
            self.tb.add_scalar('Val Acc', self.val_accuracy, self.epoch_count)
            self.tb.add_scalar('Val Loss', self.val_loss , self.epoch_count)
            self.val_duration = time.time() - self.epoch_start_time

        self.total_loss = 0
        self.total_num_correct = 0

    def end_epoch(self, images=None):
        run_duration = time.time() - self.run_start_time
        # add image and network graph to tensorboard
        if images is not None:
            grid = torchvision.utils.make_grid(images)
            self.tb.add_image('images batch', grid, dataformats='CHW')
            self.tb.add_graph(self.network, images[0:1])

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['train loss'] = self.train_loss
        results["train accuracy"] = self.train_accuracy
        results['train duration'] = self.train_duration
        results['val loss'] = self.val_loss
        results["val accuracy"] = self.val_accuracy
        results['val duration'] = self.val_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        # display = tabulate(df.values.tolist(), list(df.keys()), tablefmt='grid', missingval="?")
        # print(display)
        self.val_duration = None
        self.val_loss = None


    def track_loss(self, loss, images):
        self.total_loss += loss.item() * images.shape[0]

    def track_num_correct(self, preds, labels):
        self.total_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, filename):
        pd.DataFrame.from_dict(
            self.run_data
            ,orient='columns'
        ).to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    params = OrderedDict(
        lr = [0.1, 0.01],
        loss = ['entropy', 'dice']
        )
    runs = RunBuilder.get_runs({})
