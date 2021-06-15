# -*- coding: utf-8 -*-
import _init_paths
import os
import time
import cv2
import argparse
import torch

from lib.utils import utils, vis, get_cfg #TODO
from lib.pose_estimation.trtpose import TrtPose
from lib.tracker.deepsort import DeepSort
from lib.utils.lib_classifier import MultiPersonClassifier
from lib.utils.video import Video

def parser():
    ap = argparse.ArgumentParser()
    # options
    ap.add_argument('--mode', choices=['track', 'action'], default='track',
                    help='test mode, action recognition or tracking')
    # inference config file
    ap.add_argument("--config_file", type=str, help='inference config file',
                    default="../configs/inference.yaml")
    # input sources
    ap.add_argument('--webcam', action='store_true',
                    help="take inputs from webcam")
    ap.add_argument('--source', help='input video file path',
                    default='/home/zmh/hdd/Test_Videos/Tracking/aung_la_fight_cut_1.mp4')
    # visualization info
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    help='draw keypoints numbers info of each person',
                    default=False)
    # save folder
    ap.add_argument('--save', type=str, help='save folder',
                    default='../output')

    return ap.parse_args()


def main():
    t0 = time.time()
    args = parser()
    cfg = get_cfg(args.config_file)

    save_filename = os.path.join(
        args.save_folder, '{}_{}_{}_{}.avi'.format(
            'webcam' if args.webcam else os.path.basename(args.source[:-4]),
            cfg.TRACKER.deepsort.reid_net, args.mode, cfg.TRTPOSE.size
            ))

    # Initiate video/webcam
    if args.webcam:
        video = Video(camera=0, output_path=save_filename)
    else:
        video = Video(input_path=args.source, output_path=save_filename)

    # Initiate trtpose, deepsort and action classifier
    trtpose = TrtPose(**cfg.TRTPOSE, **cfg.TRT_CFG)
    tracker = DeepSort(**cfg.TRACKER.deepsort)
    if args.mode == 'action':
        classifier = MultiPersonClassifier(**cfg.CLASSIFIER)

    for frame in video:
        pass



if __name__ == '__main__':
    main()