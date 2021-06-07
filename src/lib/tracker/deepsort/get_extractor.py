import os
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from .models import get_model

model_root = '/home/zmh/hdd/Custom_Projects/action_recognition/my_action_recogn_dev/weights/tracker/deepsort'


class Extractor(object):
    def __init__(self, reid_net, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net = self.load_model(reid_net)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_model(self, reid_net):
        model_path = os.path.join(model_root, f'{reid_net}_reid.pth')
        print(f'[INFO] Loading deepsort reid model : {model_path}')
        net = get_model(reid_net, reid=True)
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict['net_dict'])
        net.to(self.device).eval() # add eval() for ineference mode
        return net

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)