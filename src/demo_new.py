# import _init_paths
import os
import time
import cv2
import argparse

from lib.pose_estimation import get_pose_estimator
from lib.tracker import get_tracker
from lib.classifier import get_classifier
from lib.utils import utils, vis
from lib.utils.config import Config



def get_args():
    ap = argparse.ArgumentParser()
    # configs
    ap.add_argument('--mode', choices=['track', 'action'], default='track',
                    help='inference mode for action recognition or tracking')

    ap.add_argument("--config", type=str,
                    default="../configs/inference_pipeline.yaml",
                    help='all inference configs for full action recognition pipeline.')
    # inference source
    ap.add_argument('--src',
                    default='/home/zmh/hdd/Test_Videos/Tracking/aung_la_fight_cut_1.mp4',
                    help='input source for pose estimation, video')
    # save path and visualization info
    ap.add_argument('--save_folder', type=str, default='../output',
                    help='output folder')
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    help='draw keypoints numbers of each person')
    ap.add_argument('--debug', action='store_true',
                    help='show and save for tracking bbox and detection bbox')

    return ap.parse_args()

def main():
    pass

if __name__ == '__main__':
    main()

    # Configs
    args = get_args()
    cfg = Config(args.config)
    t0 = time.time()

    # Initiate video/webcam