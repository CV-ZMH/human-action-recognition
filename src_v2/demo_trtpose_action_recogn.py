# -*- coding: utf-8 -*-
import sys
import os
import time
import cv2
import argparse
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from utils_v2 import utils, vis, parser
from pose_estimation import TrtPose
from tracking import DeepSort
from classifier import MultiPersonClassifier
# import myutils

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_deepsort", type=str,
                        default="../configs/deepsort.yaml")
    ap.add_argument('--config_trtpose', type=str,
                        default='../configs/trtpose.yaml')
    ap.add_argument('--config_classifier', type=str,
                        default='../configs/classifier.yaml')
    ap.add_argument('--src', help='input file for pose estimation, video or webcam',
                    default='/home/zmh/hdd/Test_Videos/Tracking/fun_theory_2.mp4')

    ap.add_argument('--pair_iou_thresh', type=float,
                    help='iou threshold to match with tracking bbox and skeleton bbox',
                    default=0.5)
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    help='draw keypoints numbers info of each person',
                    default=True)
    ap.add_argument('--save_path', type=str, help='output folder',
                    default='../output')
# =============================================================================
#     ap.add_argument('--add_feature_template', action='store_true',
#                     help='whether add or not feature template in top right corner',
#                     default=False)
# =============================================================================
    return ap.parse_args()


def main():
    # Configs
    args = get_args()
    cfg = parser.YamlParser()
    cfg.merge_from_file(args.config_deepsort)
    cfg.merge_from_file(args.config_trtpose)
    cfg.merge_from_file(args.config_classifier)

    # Initiate video/webcam
    cap = cv2.VideoCapture(args.src)
    assert cap.isOpened(),  f"Can't open video : {args.src}"
    filename = os.path.basename(args.src)

    # Initiate trtpose, deepsort and action classifier
    trtpose = TrtPose(**cfg.TRTPOSE)
    deepsort = DeepSort(**cfg.DEEPSORT)
    classifier = MultiPersonClassifier(**cfg.CLASSIFIER)

    frame_cnt = 0
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(filename, 640, 480)
    t0 = time.time()
    # loop on captured frames
    while True:
        ret, img_bgr = cap.read()
        if not ret: break
        img_disp = img_bgr.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # predict keypoints
        start_pose = time.time()
        trtpose_keypoints = trtpose.predict(img_rgb)
        trtpose_keypoints = trtpose.remove_persons_with_few_joints(
                                                    trtpose_keypoints,
                                                    min_total_joints=5,
                                                    min_leg_joints=0,
                                                    include_head=True)
        # change trtpose to openpose format
        openpose_keypoints = utils.trtpose_to_openpose(trtpose_keypoints)
        skeletons, _ = trtpose.keypoints_to_skeletons_list(openpose_keypoints)

        # get skeletons' bboxes
        bboxes = utils.get_skeletons_bboxes(openpose_keypoints, img_bgr)
        end_pose = time.time()
        if bboxes:
            # pass skeleton bboxes to deepsort
            start_track = time.time()
            xywhs = torch.as_tensor(bboxes)
            tracks = deepsort.update(xywhs, img_rgb, args.pair_iou_thresh)
            end_track = time.time()
            # classify tracked skeletons' actions
            if tracks:
                track_keypoints = {track_id: skeletons[track['kp_index']]
                               for track_id, track in tracks.items()}
                actions = classifier.classify(track_keypoints)
                # draw human pose actions info
                vis.draw_action_recognition(img_disp, tracks, trtpose_keypoints, actions)

        if frame_cnt == 0 and args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            os.makedirs(args.save_path, exist_ok=True)
            save_name = os.path.join(
                args.save_path,
                os.path.basename(args.src[:-4]) + '_trtpose_deepsort_action.avi'
                )
            writer = cv2.VideoWriter(
                save_name, fourcc, 30.0,
                (img_disp.shape[1], img_disp.shape[0])
                )
            print(f'Saved video file at {save_name}')

        vis.draw_frame_info(
            img_disp,
            frame=frame_cnt,
            track=len(tracks) if bboxes else 0
            )

        frame_cnt += 1
        if args.save_path:
            writer.write(img_disp)

        cv2.imshow(filename, img_disp)
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break

    print('Done. (%.3fs)' % (time.time() - t0))
    if args.save_path:
        writer.release()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

