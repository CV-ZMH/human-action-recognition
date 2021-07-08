import os
from tqdm import tqdm
import torch
from torch.backends import cudnn
from torchvision import transforms
from .deepsort.tracker_utils import get_gaussian_mask


class Trainer:
    def __init__(self, runner, train_loader, val_loader, model, optimizer, loss_fn, gpu=0, verbose=True):
        self.runner = runner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.verbose = verbose

        # setup device
        self.device = self.setup_device(gpu)

        # for data preprocessing
        self.norm = transforms.Normalize(runner.data_meta.mean, runner.data_meta.std)
        self.mask = get_gaussian_mask(*runner.data_meta.img_size) \
            if runner.run_params.gaussian_mask else None #TODO run params

        # setup training variables
        self.start_epoch = 0
        self.start_lr = 0.
        self.best_score = 0.

    def setup_device(self, device_id):
        device = torch.device(f'cuda:{device_id}' if device_id >= 0
                      and torch.cuda.is_available() else 'cpu')
        # setup cudnn settings
        if device.type != 'cpu':
            cudnn.benchmark = True
            cudnn.deterministic = True
            cudnn.enabled = True
        if self.verbose: print(f'[INFO] Running on device -> {device}')
        return device

    def save_checkpoint(self, filename, epoch, accuracy):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if self.verbose: print(f'\n[INFO] Saving checkpoint : {os.path.basename(filename)}')
        checkpoints = {
            'net_dict': self.model.state_dict(),
            'lr' : self.optimizer.param_groups[0]['lr'],
            'optim_dict': self.optimizer.state_dict(),
            'epoch' : epoch,
            'acc' : accuracy,
            }
        torch.save(checkpoints, filename)

    def load_checkpoint(self, pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        self.model.load_state_dict(checkpoint['net_dict'])
        self.start_lr = checkpoint['lr']
        self.start_epoch = checkpoint['epoch']
        self.best_score = checkpoint['acc']

        self.optimizer.load_state_dict(checkpoint['optim_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def update_optimizer_lr(self): #TODO
        for param in self.optimizer.param_groups:
            param['lr'] = self.start_lr

    def preprocess(self, images, labels):
        labels = labels.to(self.device) if len(labels) > 0 else None
        if not isinstance(images, (tuple, list)):
            images = (images,)
        images = tuple(img.to(self.device) for img in images)
        # print('mask device', self.mask.is_cuda)
        # print('img0 device', images[0].is_cuda)
        if self.mask is not None:
            self.mask = self.mask.to(self.device)
            images = tuple(img * self.mask for img in images)
        images = tuple(self.norm(img) for img in images)
        return images, labels

    def fit(self, total_epoch, pretrained):
        # prepare model for training
        # self.model.to(self.device)
        if os.path.isfile(pretrained):
            self.load_checkpoint(pretrained)
            self.update_optimizer_lr()
        # start training
        desc_template = "Epoch: [{}/{}] Iter: [{}/{}] LR: {} Loss: {:.4f}"
        for epoch in range(self.start_epoch, total_epoch):
            self.start_lr = self.optimizer.param_groups[0]['lr']
            self.model.train().to(self.device)
            self.runner.begin_dataiter() #
            tq = tqdm(enumerate(self.train_loader))
            for i_iter, (imgs, lbls) in tq:
                images, labels = self.preprocess(imgs, lbls)
                # try:
                outputs = self.model(*images)
                #calculate loss
                if labels is not None: # normal classifier
                    loss = self.loss_fn(outputs, labels)
                    self.runner.track_metric(outputs, labels)
                    self.runner.track_loss(loss, labels)
                else: # siamese triplet
                    loss = self.loss_fn(*outputs)
                    self.runner.track_loss(loss, images[0])

                self.optimizer.zero_grad() # clean gradient
                loss.backward() # calculate gradient
                self.optimizer.step() # update weights
                tq.set_description(desc=desc_template.format(
                    epoch+1, total_epoch,
                    i_iter+1, len(self.train_loader),
                    self.start_lr, loss.item()
                    ))
                del outputs, images
                torch.cuda.empty_cache()
            self.runner.end_dataiter(self.train_loader, self.model, epoch+1, prefix_tag='train')
            # start validation
            self.test(self.val_loader)
            self.runner.end_dataiter(self.val_loader, self.model, epoch+1, prefix_tag='val')
            if labels is not None:
                if self.val_acc > self.best_score:
                    self.best_score = self.val_acc
                    checkpoint_file = os.path.join(
                        self.runner.save_root, 'checkpoints', f'{self.runner.run_params}',
                        f'epoch_{epoch+1}-best_acc_{100*self.best_score}.pth')
                    self.save_checkpoint(checkpoint_file, epoch, self.best_score) #
            elif epoch % 10 == 0:
                checkpoint_file = os.path.join(
                    self.runner.save_root, 'checkpoints', f'{self.runner.run_params}',
                    f'epoch_{epoch+1}-loss_{self.val_loss}.pth')
                self.save_checkpoint(checkpoint_file, epoch, self.best_score) #

    @torch.no_grad()
    def test(self, loader):
        self.model.eval().to(self.device)
        print('Validation...')
        for images, labels in tqdm(loader):
            images, labels = self.preprocess(images, labels)
            outputs = self.model(*images)
            if labels is not None: # normal classifier
                loss = self.loss_fn(outputs, labels)
                self.runner.track_metric(outputs, labels)
                self.runner.track_loss(loss, labels)
            else: # siamese triplet
                loss = self.loss_fn(*outputs)
                self.runner.track_loss(loss, images[0])
        desc = 'Val Loss: {:.4f} Val Acc: {}'
        self.val_loss = self.runner.total_loss / len(loader.dataset) #
        self.val_acc = self.runner.total_accuracy / len(loader.dataset)  #
        print(desc.format(self.val_loss, self.val_acc)) #
