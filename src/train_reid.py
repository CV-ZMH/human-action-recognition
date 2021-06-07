import argparse

# import _init_paths
import torch
from torch.utils.data import DataLoader

from lib.utils.parser import YamlParser
from lib.tracker.deepsort import Trainer, Runner, RunBuilder
from lib.tracker.deepsort.datasets import Market1501, SiameseTriplet
from lib.tracker.deepsort.models import get_model
from lib.tracker.deepsort.loss import TripletLoss
from lib.tracker.deepsort.utils import get_transforms

def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', type=str, help='train config file path',
                    default='../configs/training_reid.yaml')
    return ap.parse_args()

def get_dataloaders(reid_net, data_path, img_size, batch_size, workers, **kwargs):
    tfms = get_transforms(*img_size)
    # load training set
    if reid_net == 'wideresnet':
        train_dataset = Market1501(data_path, mode='train', tfms=tfms)
        val_dataset = Market1501(data_path, mode='val', tfms=tfms)
    elif reid_net == 'siamesenet':
        train_dataset = SiameseTriplet(data_path, mode='train', tfms=tfms)
        val_dataset = Market1501(data_path, mode='val', tfms=tfms)

    train_loader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=batch_size,
        num_workers=workers)
    val_loader = DataLoader(
        val_dataset, shuffle=False,
        batch_size=batch_size,
        num_workers=workers)
    num_classes = train_dataset.num_classes

    return train_loader, val_loader, num_classes


def main():
    args = parser()
    cfg = YamlParser(args.config_file)
    train_meta = cfg.TRAIN.fixed_params
    data_meta = cfg.DATASET

    # load datasets
    train_loader, val_loader, num_classes = get_dataloaders(
        reid_net=train_meta.reid_net, **data_meta
        )

    runs = RunBuilder.get_runs(cfg.TRAIN.tune_params)
    runner = Runner(train_meta=train_meta, data_meta=data_meta)
    for run in runs:
        print(f'\n[INFO] {run}\n')
        runner.begin_run(run)

        # build model
        model = get_model(train_meta.reid_net, num_classes, reid=False)

        # loss
        if train_meta.reid_net == 'wideresnet':
            loss_fn = torch.nn.CrossEntropyLoss()
        elif train_meta.reid_net == 'siamesenet':
            loss_fn = TripletLoss()

        # optimizer
        if run.optim == 'SGD': #TODO getattr
            optimizer = torch.optim.SGD(
                model.parameters(), run.lr,
                momentum=0.9, weight_decay=5e-4)
        elif run.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), run.lr)

        # scheduler
        if train_meta.scheduler == 'reduce_lr':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=5, verbose=True
                )
        # trainer
        trainer = Trainer(
            runner, train_loader, val_loader, model,
            optimizer, loss_fn, gpu=train_meta.gpu, verbose=True
            )

        # training epoch
        trainer.fit(train_meta.total_epoch, train_meta.pretrained)
        if train_meta.scheduler == 'reduce_lr':
            scheduler.step(trainer.val_loss)

        runner.end_run()
        trainer.best_score = 0

if __name__ == '__main__':
    main()