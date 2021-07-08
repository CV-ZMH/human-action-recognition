# -*- coding: utf-8 -*-

import _init_paths
import time
import argparse
import os.path as osp

import cv2
import numpy as np
from pose_estimation import get_pose_estimator
from tracker import get_tracker
from classifier import get_classifier
from utils.config import Config
from utils.video import Video
from utils import utils, drawer, commons as comm


def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # configs
    ap.add_argument('--task', choices=['pose', 'track', 'action'], default='pose',
                    help='inference task for pose estimation, action recognition or tracking')

    ap.add_argument("--config", type=str,
                    default="../configs/infer_trtpose_deepsort_dnn.yaml",
                    help='all inference configs for full action recognition pipeline.')
    # inference source
    ap.add_argument('--source',
                    default='/home/zmh/Desktop/HDD/Workspace/dev/human-action-recognition/raw_imgs/original.png',
                    help='input source for pose estimation, if None, it wiil use webcam by default')
    # save path and visualization info
    ap.add_argument('--save_folder', type=str, default='../output',
                    help='just need output folder, not filename. if None, result will not save.')
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    default=True,
                    help='draw keypoints numbers of each person')

    return ap.parse_args()

def draw_keypoints_on_canvas(img, keypoints_list, thickness=2):
    image = np.zeros_like(img, dtype=np.uint8)
    visibilities = []
    centers = {}
    # draw points on image
    for keypoints in keypoints_list:
        for kp in keypoints:
            if kp[1]==0 or kp[2]==0:
                visibilities.append(kp[0])
                continue
            center = int(kp[1] * image.shape[1] + 0.5) , int(kp[2] * image.shape[0] + 0.5)
            centers[kp[0]] = center
# =============================================================================
#             cv2.circle(image, center, thickness, (0,0,255), thickness+10)
#             cv2.putText(image, str(int(kp[0])), center, cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                     0.8 if thickness==1 else 1, (255,255,255), 1)
# =============================================================================

         # draw line on image
        for pair_idx, pair in enumerate(comm.limb_pairs):
            if pair[0] in visibilities or pair[1] in visibilities: continue
            cv2.line(image, centers[pair[0]], centers[pair[1]], (0, 0, 255), 3)
            cv2.line(img, centers[pair[0]], centers[pair[1]], (0, 0, 255), 3)

        for idx, center in centers.items():
            cv2.circle(image, center, thickness, (0,255,0), thickness+10)
            # cv2.putText(image, str(int(idx)), (center[0]-10, center[1]-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            #        1, (255,255,255), thickness)
            cv2.circle(img, center, thickness, (0,255,0), thickness+10)
            # cv2.putText(img, str(int(idx)), (center[0]-10, center[1]-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            #        1, (255,255,255), thickness)
        return img, image


def main():
     # Configs
    args = get_args()
    cfg = Config(args.config)
    pose_kwargs = cfg.POSE
    source = args.source
    pose_estimator = get_pose_estimator(**pose_kwargs)
    bgr_frame = cv2.imread(source)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    # predict pose estimation
    # start_pose = time.time()
    keypoints = pose_estimator.predict(rgb_frame)
    # end_pose = time.time() - start_pose
    out_frame, kp_frame = draw_keypoints_on_canvas(bgr_frame, keypoints, thickness=2)

    cv2.imshow('overlay', out_frame)
    cv2.imshow('kp', kp_frame)
    cv2.imwrite('trtpose_kp.jpg', kp_frame)
    cv2.imwrite('trtpose_ovelay.jpg', out_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
