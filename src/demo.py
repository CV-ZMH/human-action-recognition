# -*- coding: utf-8 -*-
import _init_paths
import os
import time
import cv2
import argparse

from utils import utils, vis
from utils.config import Config
from pose_estimation.trtpose.trtpose import TrtPose
from tracker.deepsort.deepsort import DeepSort
from classifier.dnn_classifier.classifier import MultiPersonClassifier


def get_args():
    ap = argparse.ArgumentParser()
    # configs
    ap.add_argument('--mode', choices=['track', 'action'], default='track',
                    help='inference mode for action recognition or tracking')

    ap.add_argument("--config_infer", type=str,
                    default="../configs/inference_config.yaml",
                    help='deepsort config file path',
                    )
    ap.add_argument("--config_trtpose", type=str,
                    default="../configs/trtpose.yaml",
                    help='trtpose config file path',
                    )
    # inference source
    ap.add_argument('--src',
                    default='/home/zmh/hdd/Test_Videos/Tracking/aung_la_fight_cut_1.mp4',
                    help='input file for pose estimation, video or webcam',
                    )
    # thresholds for better result of tracking and action recognition
    ap.add_argument('--min_joints', type=int,
                    default=8,
                    help='minimun keypoint number threshold to use tracking and action recognition.',
                    )
    ap.add_argument('--min_leg_joints', type=int,
                    default=3,
                    help='minimun legs keypoint number threshold to use tracking and action recogniton.',
                    )
    # save path and visualization info
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    default=False,
                    help='draw keypoints numbers info of each person',
                    )
    ap.add_argument('--save_folder', type=str,
                    default='../output',
                    help='output folder',
                    )
    ap.add_argument('--debug', action='store_true',
                    default=True,
                    help='show and save for tracking bbox and detection bbox',
                    )
# =============================================================================
#     ap.add_argument('--add_feature_template', action='store_true',
#                     help='whether add or not feature template in top right corner',
#                     default=False)
# =============================================================================
    return ap.parse_args()


def main():
    ## Configs
    args = get_args()
    cfg = Config()
    cfg.merge_from_file(args.config_infer)
    cfg.merge_from_file(args.config_trtpose)
    t0 = time.time()

    ## Initiate video/webcam
    cap = cv2.VideoCapture(args.src if args.src else 0)
    assert cap.isOpened(),  f"Can't open video : {args.src}"
    display_name = os.path.basename(args.src) if args.src else 'webcam'

    try:
        ## Initiate trtpose, deepsort and action classifier
        trtpose = TrtPose(**cfg.TRTPOSE, ** cfg.TRTPOSE.trt_model)
        tracker = DeepSort(**cfg.TRACKER.deepsort)
        if args.mode == 'action':
            classifier = MultiPersonClassifier(**cfg.CLASSIFIER)

        frame_cnt = 0
        if args.debug:
            cv2.namedWindow('debug', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('debug', 640, 480)
            debug_img = None

        cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(display_name, 960, 720)
        cv2.moveWindow(display_name, 20, 20)

        ## loop on captured frames
        while True:
            ret, img_bgr = cap.read()
            if not ret: break
            img_disp = img_bgr.copy()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ## predict keypoints
            start_pose = time.time()
            counts, objects, peaks = trtpose.predict(img_rgb)
            trtpose_keypoints = trtpose.get_keypoints(objects, counts, peaks)
            trtpose_keypoints = trtpose.remove_persons_with_few_joints(
                                                        trtpose_keypoints,
                                                        min_total_joints=args.min_joints,
                                                        min_leg_joints=args.min_leg_joints,
                                                        include_head=True)
            ## change trtpose to openpose format
            openpose_keypoints = utils.trtpose_to_openpose(trtpose_keypoints)
            skeletons, _ = trtpose.keypoints_to_skeletons_list(openpose_keypoints)
            ## get skeletons' bboxes
            bboxes = utils.get_skeletons_bboxes(openpose_keypoints, img_rgb)

            # end_pose = time.time() - start_pose
            if len(bboxes) > 0:
                ## pass skeleton bboxes to deepsort
                start_track = time.time()
                tracks, debug_img = tracker.update(bboxes, img_rgb, debug=args.debug)
                end_track = time.time() - start_track

                ## classify tracked skeletons' actions
                if tracks is not None:
                    if args.mode == 'action':
                        # start_action = time.time()
                        track_keypoints = {track["track_id"]: skeletons[track["detection_index"]]
                                       for track in tracks}
                        actions = classifier.classify(track_keypoints)
                        # end_action = time.time() - start_action

                    ## draw result info to image
                    vis.draw_frame(
                        img_disp, tracks, trtpose_keypoints,
                        draw_numbers=args.draw_kp_numbers,
                        actions=actions if args.mode=='action' else None
                        )

            else: # if no detection, update tracked's time_since_update and it's age
                tracker.increment_ages()

            end_total = time.time() - start_pose
            ## draw information of the current frame
            final_img = vis.draw_frame_info(
                    img_disp,
                    color='green',
                    frame=frame_cnt,
                    mode=args.mode,
                    max_dist=cfg.TRACKER.deepsort.max_dist,
                    max_iou_dist=cfg.TRACKER.deepsort.max_iou_distance,
                    Speed='{:.3f}s'.format(end_total),
                    # TrackSpeed='{:.3f}s'.format(end_track) if len(bboxes)>0 else 0,
                    Tracks=len(tracks) if len(bboxes) > 0 else 0,
                )

            ## write video
            if frame_cnt == 0 and args.save_folder:
               os.makedirs(args.save_folder, exist_ok=True)
               fourcc = cv2.VideoWriter_fourcc(*'XVID')
               save_name = os.path.join(
                               args.save_folder, '{}_trtpose_{}_{}_{}.avi'.format(
                                   os.path.basename(args.src[:-4]) if args.src else 'webcam',
                                   cfg.TRACKER.deepsort.reid_net,
                                   args.mode, cfg.TRTPOSE.size
                               ))

               writer = cv2.VideoWriter(
                   save_name, fourcc, 20.0,
                   (final_img.shape[1], final_img.shape[0])
                   )
               if args.debug:
                   debug_writer = cv2.VideoWriter(
                       save_name[:-4] + '_debug.avi', fourcc, 20.0,
                       (final_img.shape[1], final_img.shape[0])
                       )
            frame_cnt += 1
            if args.save_folder:
                writer.write(final_img)
                if args.debug:
                    if debug_img is None:
                        debug_img = img_disp[...,::-1].copy()
                    debug_writer.write(debug_img[...,::-1])

            # show frame
            cv2.imshow(display_name, final_img)
            if args.debug:
                if debug_img is None:
                    debug_img = img_disp[...,::-1].copy()
                cv2.imshow('debug', debug_img[...,::-1])

            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break

    finally:
        print('Finished in : %.3fs' % (time.time() - t0))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()