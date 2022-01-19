'''
Load skeleton data from `skeletons_info.txt`,
process data,
and then save features and labels to .csv file.
'''
import _init_paths
import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate
from utils import utils
from utils.config import Config


def skeleton_loader(files):
    for file in files:
        with open(file, 'r') as f:
            skeleton_data = json.load(f)
        yield skeleton_data

def main():
    # Settings
    cfg = Config(config_file='../configs/train_action_recogn_pipeline.yaml')
    cfg_state = cfg[os.path.basename(__file__)]

    ## IO folders
    get_path = lambda x: os.path.join(*x) if isinstance(x, (list, tuple)) else x
    skeletons_folder = get_path(cfg_state.input.skeletons_folder)
    skeletons_txt = get_path(cfg_state.output.skeletons_txt)

    ## Config for training
    idx_person = 0  # Only use the skeleton of the 0th person in each image
    idx_label = 3  # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
    classes = np.array(cfg.classes)

    # Get skeleton files
    files = utils.get_files(skeletons_folder, extensions='.txt')
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

    print(tabulate([list(labels_cnt.values())],
                   list(labels_cnt.keys()), 'grid'))
    print(f'[INFO] Total numbers of combined skeletons: "{len(all_skeletons)}"')

if __name__ == "__main__":
    main()
