# -*- coding: utf-8 -*-
import numpy as np
from .commons import *

def trtpose_to_openpose(trtpose_keypoints):
    """Change trtpose skeleton to openpose format"""

    new_keypoints = trtpose_keypoints.copy()

    if new_keypoints.tolist():
        for idx1, idx2 in openpose_trtpose_match_idx:
            new_keypoints[:, idx1, 1:] = trtpose_keypoints[:, idx2, 1:] # neck
    return new_keypoints

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
    x_center = new_xmin + (new_width/2)
    y_center = new_ymin + (new_height/2)

    return [int(x_center), int(y_center), int(new_width), int(new_height)]

def get_skeletons_bboxes(all_keypoints, image):
    """Get list of (xcenter, ycenter, width, height) bboxes from all persons keypoints"""

    bboxes = []
    img_h, img_w =  image.shape[:2]
    for idx, keypoints in enumerate(all_keypoints):
        keypoints = np.where(keypoints[:, 1:] !=0, keypoints[:, 1:], np.nan)
        keypoints[:, 0] *= img_w
        keypoints[:, 1] *= img_h
        xmin = np.nanmin(keypoints[:,0])
        ymin = np.nanmin(keypoints[:,1])
        xmax = np.nanmax(keypoints[:,0])
        ymax = np.nanmax(keypoints[:,1])
        bbox = expand_bbox(xmin, xmax, ymin, ymax, img_w, img_h)
        bboxes.append(bbox)

    return bboxes

