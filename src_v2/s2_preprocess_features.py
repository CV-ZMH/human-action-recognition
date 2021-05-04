#!/usr/bin/env python
# coding: utf-8

'''
Load skeleton data from `skeletons_info.txt`,
process data,
and then save features and labels to .csv file.
'''
import sys
import os
import json
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from classifier.feature_procs import extract_multi_frame_features
from utils_v2 import commons, parser, utils
import myutils

# Settings
cfg = parser.YamlParser(config_file='../configs/pipeline_trtpose.yaml')
cfg.merge_from_file('../configs/trtpose.yaml')
cfg_state = cfg[os.path.basename(__file__)]

## IO folders
skeletons_folder = os.path.join(*cfg_state.input.skeletons_folder)
features_x_path = os.path.join(*cfg_state.output.features_x)
features_y_path = os.path.join(*cfg_state.output.features_y)

## Config for training
IDX_PERSON = 0  # Only use the skeleton of the 0th person in each image
IDX_ACTION_LABEL = 3  # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
window_size = cfg.window_size
classes = np.array(cfg.classes)


def process_features(X0, Y0, video_indices, classes, window_size):
    ''' Process features '''
    # Convert features
    # From: raw feature of individual image.
    # To:   time-serials features calculated from multiple raw features
    #       of multiple adjacent images, including speed, normalized pos, etc.
    ADD_NOISE = False
    if ADD_NOISE:
        X1, Y1 = extract_multi_frame_features(
            X0, Y0, video_indices, window_size,
            is_adding_noise=True, is_print=True)
        X2, Y2 = extract_multi_frame_features(
            X0, Y0, video_indices, window_size,
            is_adding_noise=False, is_print=True)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        X, Y = extract_multi_frame_features(
            X0, Y0, video_indices, window_size,
            is_adding_noise=False, is_print=True)
        return X, Y

#%% s3 dataloader
def skeleton_loader(files):
    for file in files:
        with open(file, 'r') as f:
            skeleton_data = json.load(f)
        yield skeleton_data



# -- Main
def main():
    pass

if __name__ == "__main__":
    main()
    # Get skeleton files
    files = myutils.get_files(skeletons_folder, extensions='.txt')
    data_loader = skeleton_loader(files)

