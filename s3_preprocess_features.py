# -*- coding: utf-8 -*-
import os
import numpy as np

from utils import lib_commons
from utils.lib_skeletons_io import load_skeleton_data
from utils.lib_feature_proc import extract_multi_frame_features

def process_features(x0, y0, video_indices, classes, add_noise=False, window_size=5):
    """ Process features """
    if add_noise:
        x1, y1 = extract_multi_frame_features(
            x0, y0, video_indices, window_size,
            is_adding_noise=True)
        x2, y2 = extract_multi_frame_features(
            x0, y0, video_indices, window_size,
            is_adding_noise=False)
        x = np.vstack((x1, x2))
        y = np.concatenate((y1, y2))
    else:
        x, y = extract_multi_frame_features(
            x0, y0, video_indices, window_size,
            is_adding_noise=False)
    return x, y


def main():
    # -- setting
    cfg_all = lib_commons.read_yaml('config/config.yaml')
    cfg = cfg_all[os.path.basename(__file__)]

    # action recogn
    ADD_NOISE = False
    WINDOW_SIZE = int(cfg_all['features']['window_size']) # num of frames used to extract features

    classes = cfg_all['classes']
    src_all_skeletons_txt = cfg['input']['all_skeletons_txt']
    dst_processed_features = cfg['output']['processed_features']
    dst_processed_features_labels = cfg['output']['processed_features_labels']

    x0, y0, video_indices = load_skeleton_data(src_all_skeletons_txt, classes, add_noise=ADD_NOISE, window_size=WINDOW_SIZE)
    # process features
    print('\nExtacting time-serial features...')
    x, y = process_features(x0, y0, video_indices, classes)
    print(f'X.shape = {x.shape}, len(Y) = {len(y)}')

    # save data
    print('\nWriting features and labels to disk ...')
    os.makedirs(os.path.dirname(dst_processed_features), exist_ok=True)
    os.makedirs(os.path.dirname(dst_processed_features_labels), exist_ok=True)

    np.savetxt(dst_processed_features, x, fmt='%.5f')
    print(f'Save features to : {dst_processed_features}')

    np.savetxt(dst_processed_features_labels, y, fmt='%i')
    print(f'Save labels to : {dst_processed_features_labels}')

if __name__ == '__main__':
    main()