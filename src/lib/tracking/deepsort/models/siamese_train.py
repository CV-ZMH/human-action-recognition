import sys
sys.path.insert(0, '../../')
import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from siamese.siamese_dataset import SiameseTriplet, get_gaussian_mask
from siamese.siamese_net import SiameseNetwork, TripletLoss
from utils import parser

# CUDNN related setting
cudnn.benchmark = True
cudnn.enabled = True
cudnn.deterministic = True


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", type=str, default="../../../../configs/training_siamese.yaml",
                    help='training siamese config file path')
    args = ap.parse_args()

    # load training configs
    cfg = parser.YamlParser(config_file=args.train_config)
    return cfg



class Trainer:
    def __init__(self):
        self.cfg = get_parser()

    def train(self):
        pass

if __name__ == '__main__':
    trainer = Trainer()
