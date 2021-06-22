import os
import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms
from .get_reid import get_reid_network
from .utils import get_gaussian_mask, show_tensor


class FeatureExtractor(object):
    def __init__(self, reid_net, model_path, img_size, mean, std, gaussian_mask=False, use_cuda=True, **kwargs): #TODO gaussian mask
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.reid_net = reid_net
        if isinstance(model_path, (tuple, list)):
            model_path = os.path.join(*model_path)
        self.model = self.load_model(model_path)
        self.mask = None

        if gaussian_mask:
            self.mask = get_gaussian_mask(*img_size)
        self.tfms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            ])
        self.norm = transforms.Normalize(mean, std)

    def load_model(self, model_path):
        assert os.path.isfile(model_path), f"reid model don't exit : {model_path}"

        print(f'[INFO] Loading deepsort reid model : {model_path}')
        model = get_reid_network(self.reid_net, num_classes=751, reid=True)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['net_dict'])
        model.to(self.device).eval() # add eval() for ineference mode
        return model

    def _preprocess(self, img_crops):
        if self.mask is not None:
            mask_transform = lambda x: self.norm(self.tfms(Image.fromarray(x))*self.mask)
            imgs = [mask_transform(img)[None] for img in img_crops]
        else:
            no_mask_transform = lambda x: self.norm(self.tfms(Image.fromarray(x)))
            imgs = [no_mask_transform(img)[None] for img in img_crops]

        img_batch = torch.cat(imgs, dim=0).float()
        return img_batch.to(self.device)

    @torch.no_grad()
    def __call__(self, img_crops):
        img_batch = self._preprocess(img_crops)
        if self.reid_net == 'siamesenet':
            features = self.model.forward_once(img_batch)
            features.squeeze_(-1).squeeze_(-1)
            features.div_(features.norm(p=2, dim=1, keepdim=True))

        elif self.reid_net == 'wideresnet':
            features = self.model(img_batch)

        elif self.reid_net == 'align_reid':
            features, _ = self.model(img_batch)
            features.div_(features.norm(p=2, dim=1, keepdim=True))

        return features.cpu().numpy()