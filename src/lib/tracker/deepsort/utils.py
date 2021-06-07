# -*- coding: utf-8 -*-
from tqdm import tqdm
import cv2
from PIL import Image

import torch
import numpy as np
from torchvision import transforms
from scipy.stats import multivariate_normal


def show_tensor(tensor):
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    cv2.namedWindow('display', cv2.WND_PROP_FULLSCREEN)
    try:
        cv2.imshow('display', np_img[...,::-1])
        cv2.waitKey(0)
    except Exception as e:
        print(f'ERROR {e}')
    finally:
        cv2.destroyAllWindows()

def get_transforms(H, W):
    tfms = transforms.Compose([
        transforms.Resize((H, W)), # h,w
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=Image.BILINEAR),
        transforms.ToTensor(),
        ])
    return tfms

def normalize(x, mean, std):
    return (x - mean).div(std)

def get_gaussian_mask(H, W):
	#128 is image size
	# We will be using 256x128 patch instead of original 128x128 path because we are using for pedestrain with 1:2 AR.
	x, y = np.mgrid[0:1.0:complex(H), 0:1.0:complex(W)] #128 is input size.
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.28,0.28])
	covariance = np.diag(sigma**2)
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
	z = z.reshape(x.shape)
	z = z / z.max()
	z  = z.astype(np.float32)
	mask = torch.from_numpy(z)
	return mask

def get_mean_std(loader, H, W):
    # var[x] = E[x**2] - E[x]**2
    mask = get_gaussian_mask(H, W)
    mean, var, nb_samples = 0, 0, 0
    for data in tqdm(loader):
        # we will add gaussian mask on the batches of images
        data = data.cuda() * mask.cuda()
        batch = data.size(0)
        data = data.view(batch, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        nb_samples += batch
    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    return mean, std