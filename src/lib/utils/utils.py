# -*- coding: utf-8 -*-
import os
import time
import mimetypes
import pathlib
from pathlib import Path
from typing import Iterable

import torch
import cv2
import numpy as np
from .commons import *

import myutils


to_min = lambda x: f"{int(x // 60)}m {x % 60:.3f}s"
def exec_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f'=> Speed of [{func.__name__}] func : {to_min(end-start)}')
        return res
    return inner

def stack(img):
    if isinstance(img, (str, pathlib.PosixPath)): img = cv2.imread(str(img))
    return np.stack([img]*3, axis=-1) if img.ndim == 2 else img

def show(imgs, wait=0, window='show', text=None, text_pos=[]):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().detach().squeeze().permute(1,2,0).numpy()
    elif isinstance(imgs, (str, pathlib.PosixPath)): # is path
        assert os.path.exists(imgs), f'File not exists {imgs}'
        imgs = cv2.imread(str(imgs))
    elif isinstance(imgs, (list, tuple)):
        imgs = [stack(img) for img in imgs]
        imgs = np.concatenate([*imgs], axis=1) # horizonal concat
    image = imgs.copy()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if text:
        draw_text(image, str(text), pos=text_pos)
    cv2.imshow(window, image)
    key = cv2.waitKey(wait)
    if not wait: cv2.destroyWindow(window)
    if key==27 or key==ord('q') or key==ord(' '): raise StopIteration
    if text: return image

def draw_text(image, text, pos=[], scale=1.4, size=2, color=(0,255,0)):
    if len(pos) == 0:
        h, w = image.shape[:2]
        pos = ((w//2)-80, 20)
    text_size, _ = cv2.getTextSize(text, 0, scale, size)
    cv2.line(image, pos, (pos[0]+text_size[0], pos[1]), (0,0,200), text_size[1]+14)
    cv2.line(image, pos, (pos[0]+text_size[0], pos[1]), (10,10,10), text_size[1]+10)
    cv2.putText(image, str(text), (pos[0], pos[1]+4), 0, scale, color, size)


def keypoints_to_skeletons_list(all_keypoints):
    """Get skeleton data of (x, y) from humans."""
    skeletons_list = []
    NaN = 0
    for keypoints in all_keypoints:
        skeleton = [NaN]*(18*2)
        for idx, kp in enumerate(keypoints):
            skeleton[2*idx] = kp[1]
            skeleton[2*idx+1] = kp[2]
        skeletons_list.append(skeleton)
    return skeletons_list

def trtpose_to_openpose(keypoints_list):
    """Change trtpose skeleton to openpose format"""

    new_keypoints = keypoints_list.copy()
    if new_keypoints.tolist():
        for idx1, idx2 in OPENPOSE_TO_TRTPOSE_IDXS:
            new_keypoints[:, idx1, 1:] = keypoints_list[:, idx2, 1:] # neck
    return new_keypoints

def convert_to_openpose_skeletons(predictions): # TODO move to annotation class method
    """Prepare trtpose keypoints for action recognition.
    First, convert openpose keypoints format from trtpose keypoints for
    action recognition as it's features extraction step is based on
    openpose keypoint format.
    Then, changed to skeletons list.
    """
    keypoints_list = np.array([pred.keypoints for pred in predictions])
    openpose_keypoints = trtpose_to_openpose(keypoints_list)
    NaN = 0
    for i, keypoints in enumerate(openpose_keypoints):
        skeletons = [NaN]*(18*2)
        for j, kp in enumerate(keypoints):
            skeletons[2*j] = kp[1]
            skeletons[2*j+1] = kp[2]
        predictions[i].flatten_keypoints = skeletons
    return predictions

def expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height):
    """expand bbox for containing more background"""
    width = xmax - xmin
    height = ymax - ymin
    ratio = 0.1   # expand ratio
    new_xmin = np.clip(xmin - ratio * width, 0, img_width)
    new_xmax = np.clip(xmax + ratio * width, 0, img_width)
    new_ymin = np.clip(ymin - ratio * height, 0, img_height)
    new_ymax = np.clip(ymax + ratio * height, 0, img_height)
    new_width = new_xmax - new_xmin
    new_height = new_ymax - new_ymin
    # x_center = new_xmin + (new_width/2)
    # y_center = new_ymin + (new_height/2)
    return [new_xmin, new_ymin, new_width, new_height]

def keypoints_to_bbox(keypoints_list, image):
    """Prepare bboxes from keypoints for object tracking.
    args:
        keypoints_list (np.ndarray): trtpose keypoints list
    return:
        bboxes (np.ndarray): bbox of (xmin, ymin, width, height)
    """

    bboxes = []
    img_h, img_w =  image.shape[:2]
    for idx, keypoints in enumerate(keypoints_list):
        keypoints = np.where(keypoints[:, 1:] !=0, keypoints[:, 1:], np.nan)
        keypoints[:, 0] *= img_w
        keypoints[:, 1] *= img_h
        xmin = np.nanmin(keypoints[:,0])
        ymin = np.nanmin(keypoints[:,1])
        xmax = np.nanmax(keypoints[:,0])
        ymax = np.nanmax(keypoints[:,1])
        bbox = expand_bbox(xmin, xmax, ymin, ymax, img_w, img_h)
        # discard bbox with width and height == 0
        if bbox[2] < 1 or bbox[3] < 1:
            continue
        bboxes.append(bbox)
    return np.asarray(bboxes)

# files IO
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def setify(x):
    # print(f'type {type(x)}, listify {listify(x)}')
    return x if isinstance(x, set) else set(listify(x))

def get_extensions(file_type='image'):
    return set(k for k, v in mimetypes.types_map.items() if v.startswith(file_type + '/'))

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if f is not f.startswith('.')
           and ((not extensions) or f".{f.split('.')[-1].lower()}" in extensions)] # check for sure folders
    return res

def get_files(path, extensions=None, recurse=False, include=None):
    """get all files path"""
    path = Path(path)
    extensions = setify(extensions) if isinstance(extensions, str) \
        else setify(e.lower() for e in extensions)

    if recurse:
        res = []
        for i, (p,d,f) in enumerate(os.walk(path)):
            if include is not None and i==0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(('.', '__'))]
            res += _get_files(p, f, extensions)
        return sorted(res)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return sorted(_get_files(path, f, extensions))

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
