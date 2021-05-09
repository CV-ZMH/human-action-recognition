# -*- coding: utf-8 -*-
import _init_paths
import os
import time
import cv2
import argparse
import torch
from utils import utils, vis, parser
from pose_estimation import TrtPose
from tracking import DeepSort
from classifier import MultiPersonClassifier

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['track', 'action'], default='action',
                    help='inference mode for action recognition or tracking')
    # configs
    ap.add_argument("--config_infer", type=str, default="../configs/inference_config.yaml",
                    help='deepsort config file path')
    ap.add_argument("--config_trtpose", type=str, default="../configs/trtpose.yaml",
                    help='trtpose config file path')
    # inference
    ap.add_argument('--src', help='input file for pose estimation, video or webcam',
                    default='/home/zmh/hdd/Test_Videos/Tracking/fun_theory_1.mp4')
                    # default='../test_data/aung_la.mp4')
    ap.add_argument('--pair_iou_thresh', type=float,
                    help='iou threshold to match with tracking bbox and skeleton bbox',
                    default=0.5)
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    help='draw keypoints numbers info of each person',
                    default=False)
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
    cfg.merge_from_file(args.config_infer)
    cfg.merge_from_file(args.config_trtpose)

    # Initiate video/webcam
    cap = cv2.VideoCapture(args.src)
    assert cap.isOpened(),  f"Can't open video : {args.src}"
    filename = os.path.basename(args.src) if args.src != 0 else 'webcam'

    # Initiate trtpose, deepsort and action classifier
    trtpose = TrtPose(**cfg.TRTPOSE_TRT)
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
                                                    min_total_joints=8,
                                                    min_leg_joints=4,
                                                    include_head=True)
        # change trtpose to openpose format
        openpose_keypoints = utils.trtpose_to_openpose(trtpose_keypoints)
        skeletons, _ = trtpose.keypoints_to_skeletons_list(openpose_keypoints)

        # get skeletons' bboxes
        bboxes = utils.get_skeletons_bboxes(openpose_keypoints, img_bgr)
        end_pose = time.time() - start_pose
        if bboxes:
            # pass skeleton bboxes to deepsort
            start_track = time.time()
            xywhs = torch.as_tensor(bboxes)
            tracks = deepsort.update(xywhs, img_rgb, args.pair_iou_thresh)
            end_track = time.time() - start_track

            # classify tracked skeletons' actions
            if tracks:
                if args.mode == 'action':
                    track_keypoints = {track_id: skeletons[track['kp_index']]
                                   for track_id, track in tracks.items()}
                    actions = classifier.classify(track_keypoints)

                # draw result info to image
                vis.draw_frame(
                    img_disp, tracks, trtpose_keypoints,
                    draw_numbers=args.draw_kp_numbers,
                    actions=actions if args.mode=='action' else None
                    )

        # else:
            # deepsort.increment_ages() # better tracking result without this function

        end_total = time.time() - start_pose
        if frame_cnt == 0 and args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            os.makedirs(args.save_path, exist_ok=True)
            save_name = os.path.join(
                args.save_path,
                '{}_trtpose_deepsort_{}.avi'
                .format(os.path.basename(args.src[:-4]) if args.src != 0 else 'webcam',
                         args.mode)
                )
            writer = cv2.VideoWriter(
                save_name, fourcc, 20.0,
                (img_disp.shape[1], img_disp.shape[0])
                )
            print(f'Saving video file to {save_name}')

        vis.draw_frame_info(
            img_disp,
            color='red',
            mode=args.mode,
            speed = '{:.4f}s'.format(end_total),
            frame=frame_cnt,
            track=len(tracks) if bboxes else 0,
            )

        frame_cnt += 1
        if args.save_path:
            writer.write(img_disp)

        cv2.imshow(filename, img_disp)
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break

    print('Finished in : %.3fs' % (time.time() - t0))
    if args.save_path:
        writer.release()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

