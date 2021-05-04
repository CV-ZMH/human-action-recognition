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
from tqdm import tqdm
from tabulate import tabulate

ROOT = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
sys.path.append(ROOT)

from utils_v2 import parser
import myutils

def skeleton_loader(files):
    for file in files:
        with open(file, 'r') as f:
            skeleton_data = json.load(f)
        yield skeleton_data

# -- Main
def main():

    # Settings
    cfg = parser.YamlParser(config_file='../configs/pipeline_trtpose.yaml')
    cfg.merge_from_file('../configs/trtpose.yaml')
    cfg_state = cfg[os.path.basename(__file__)]

    ## IO folders
    skeletons_folder = os.path.join(*cfg_state.input.skeletons_folder)
    skeletons_txt = os.path.join(*cfg_state.output.skeletons_txt)

    ## Config for training
    idx_person = 0  # Only use the skeleton of the 0th person in each image
    idx_label = 3  # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
    classes = np.array(cfg.classes)

    # Get skeleton files
    files = myutils.get_files(skeletons_folder, extensions='.txt')
    data_loader = skeleton_loader(files)

    all_skeletons = []
    labels_cnt = defaultdict(int)
    tq = tqdm(data_loader, total=len(files))
    for skeletons in tq:
        if not skeletons:
            continue
        skeleton = skeletons[idx_person]
        label = skeleton[idx_label]
        if label not in classes:
            continue
        labels_cnt[label] += 1
        all_skeletons.append(skeleton)

    with open(skeletons_txt, 'w') as f:
        json.dump(all_skeletons, f)

    print(f'[INFO] Total numbers of combined skeletons: "{len(all_skeletons)}"')
    print(tabulate([list(labels_cnt.values())],
                   list(labels_cnt.keys()), 'grid'))


if __name__ == "__main__":
    main()
