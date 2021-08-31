# -*- coding: utf-8 -*-
import os
import mimetypes
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
from .commons import *

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
    """Change trtpose skeleton to openpose format."""
    new_keypoints = keypoints_list.copy()
    if new_keypoints.tolist():
        for idx1, idx2 in openpose_trtpose_match_idx:
            new_keypoints[:, idx1, 1:] = keypoints_list[:, idx2, 1:] # neck
    return new_keypoints

def convert_to_skeletons(keypoints_list):
    """Prepare trtpose keypoints for action recognition.
    First, convert openpose keypoints format from trtpose keypoints for
    action recognition as it's features extraction step is based on
    openpose keypoint format.
    Then, changed to skeletons list.
    """
    openpose_keypoints = trtpose_to_openpose(keypoints_list)
    skeletons_list = []
    NaN = 0
    for keypoints in openpose_keypoints:
        skeleton = [NaN]*(18*2)
        for idx, kp in enumerate(keypoints):
            skeleton[2*idx] = kp[1]
            skeleton[2*idx+1] = kp[2]
        skeletons_list.append(skeleton)
    return skeletons_list

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
