# -*- coding: utf-8 -*-
import os
from shutil import copyfile
from tqdm import tqdm
from fire import Fire

# import _init_paths
# from utils import utils

# def copy_files(src, dst, extensions={'.jpg'}, recurse=False):
#     files = utils.get_files(src, extensions=extensions, recurse=recurse)
#     print(f'"{src}" --> {len(files)}.\n')
#     for file in tqdm(files):
#         ID = file.name.split('_')[0]
#         dst_path = os.path.join(dst, ID)
#         if not os.path.isdir(dst_path): os.makedirs(dst_path)
#         copyfile(file, os.path.join(dst_path, file.name))

def main(root):

    train_path = os.path.join(root, 'bounding_box_train')
    dst =  os.path.join(root, 'prepare_data')
    os.makedirs(dst, exist_ok=True)
    train_save_path = os.path.join(dst, 'train')
    val_save_path = os.path.join(dst, 'val')
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    for root, dirs, files in tqdm(os.walk(train_path, topdown=True)):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = os.path.join(train_path, name)
            dst_path = os.path.join(train_save_path, ID[0])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = os.path.join(val_save_path, ID[0])  #first image is used as val image
                os.mkdir(dst_path)
            copyfile(src_path, os.path.join(dst_path, name))


if __name__ == '__main__':

    Fire(main)
