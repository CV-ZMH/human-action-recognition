# -*- coding: utf-8 -*-
import sys
import os
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

import json
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict
from utils import lib_commons

def get_skeletons_length(file_paths):
    ''' Find a non-empty txt file, and then get the length of one skeleton data.
    The data length should be 41, where:
    41 = 5 + 36.
        5: [cnt_action, cnt_clip, cnt_image, action_label, filepath]
            See utils.lib_io.get_training_imgs_info for more details
        36: 18 joints * 2 xy positions
    '''
    for file_path in file_paths:
        skeletons = lib_commons.read_listlist(file_path)
        if len(skeletons):
            skeleton = skeletons[idx_person]
            data_length = len(skeleton)
            assert(data_length == 41)
            return data_length


if __name__ == '__main__':

    cfg_all = lib_commons.read_yaml(os.path.join(ROOT, 'config', 'config.yaml'))
    cfg = cfg_all[os.path.basename(__file__)]

    classes = cfg_all['classes']
    skeleton_filename_format = cfg_all['skeleton_filename_format']
    src_skeletons_folder = cfg['input']['skeletons_folder']
    dst_all_skeletons_txt = cfg['output']['all_skeletons_txt']

    idx_person = 0 # Only use one person's skeleton in each image
    idx_label = 3 # action label index in skeleton file

    file_paths = lib_commons.get_filenames(src_skeletons_folder, use_sort=True, with_folder_path=True)
    data_length = get_skeletons_length(file_paths)
    print(f'Data length of skeleton file is {data_length}')
    all_skeletons = []
    labels_cnt = defaultdict(int)

    tq = tqdm(file_paths, total=len(file_paths))
    for file_path in tq:
        skeletons = lib_commons.read_listlist(file_path)
        if not skeletons: continue
        skeleton = skeletons[idx_person]
        label = skeleton[idx_label]
        if label not in classes: continue
        labels_cnt[label] += 1
        all_skeletons.append(skeleton)

    with open(dst_all_skeletons_txt, 'w') as f:
        json.dump(all_skeletons, f)

    print(f'Combined {len(all_skeletons)}/{len(file_paths)} skeleton data.')
    print(f'Saved to {dst_all_skeletons_txt}\n')
    print(f'Total actions : {len(labels_cnt)} action')
    display = tabulate([list(labels_cnt.values())], list(labels_cnt.keys()), 'grid') # data, header, table_fmt
    print(display)




