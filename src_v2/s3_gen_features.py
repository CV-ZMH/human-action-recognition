#!/usr/bin/env python
# coding: utf-8

'''
Load skeleton data from `skeletons_info.txt`,
process data,
and then save features and labels to .csv file.
'''

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

from classifier.feature_procs import extract_multi_frame_features
from utils_v2 import parser
from utils_v2.skeletons_io import load_skeleton_data

def process_features(X0, Y0, video_indices, classes, window_size=5):
    ''' Process features '''
    # Convert features
    # From: raw feature of individual image.
    # To:   time-serials features calculated from multiple raw features
    #       of multiple adjacent images, including speed, normalized pos, etc.
    ADD_NOISE = False
    if ADD_NOISE:
        X1, Y1 = extract_multi_frame_features(
            X0, Y0, video_indices, window_size,
            is_adding_noise=True, is_print=False)
        X2, Y2 = extract_multi_frame_features(
            X0, Y0, video_indices, window_size,
            is_adding_noise=False, is_print=False)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        X, Y = extract_multi_frame_features(
            X0, Y0, video_indices, window_size,
            is_adding_noise=False, is_print=False)
        return X, Y

# -- Main


def main():
    '''
    Load skeleton data from `skeletons_info.txt`, process data,
    and then save features and labels to .csv file.
    '''
    # Settings
    cfg = parser.YamlParser(config_file='../configs/pipeline_trtpose.yaml')
    cfg_stage = cfg[os.path.basename(__file__)]
    classes = np.array(cfg.classes)
    window_size = cfg.window_size

    ## IO folder
    src_skeletons_txt = os.path.join(*cfg_stage.input.skeletons_txt)
    dst_features_X = os.path.join(*cfg_stage.output.features_x)
    dst_features_Y = os.path.join(*cfg_stage.output.features_y)

    # Load data
    X0, Y0, video_indices = load_skeleton_data(src_skeletons_txt, classes)
    print(f"X0 {len(X0)}, Y0 {len(Y0)}")
    print(f"video indices {len(video_indices)}")

    # Process features
    print("\nExtracting time-serials features ...")
    X, Y = process_features(X0, Y0, video_indices, classes, window_size)
    print(f"X.shape = {X.shape}, len(Y) = {len(Y)}")

    # Save data
    print("\nWriting features and labesl to disk ...")

    os.makedirs(os.path.dirname(dst_features_X), exist_ok=True)
    os.makedirs(os.path.dirname(dst_features_Y), exist_ok=True)

    np.savetxt(dst_features_X, X, fmt="%.5f")
    print("Save features to: " + dst_features_X)

    np.savetxt(dst_features_Y, Y, fmt="%i")
    print("Save labels to: " + dst_features_Y)


if __name__ == "__main__":
    main()
