import _init_paths
import time
import argparse
import os.path as osp

import cv2
from pose_estimation import get_pose_estimator
from tracker import get_tracker
from classifier import get_classifier
from utils.config import Config
from utils.video import Video
from utils.drawer import Drawer

from utils import utils

def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--task', choices=['pose', 'track', 'action'], default='action',
                    help='inference task for pose estimation, action recognition or tracking')

    ap.add_argument("--config", type=str,
                    default="../configs/infer_trtpose_deepsort_dnn.yaml",
                    help='all inference configs for full action recognition pipeline.')
    # inference source
    ap.add_argument('--source',
                    default='/home/zmh/Desktop/HDD/Test_Videos/fun_theory_2.mp4',
                    help='input source for pose estimation, if None, it wiil use webcam by default')

    # save path and visualization info
    ap.add_argument('--save_folder', type=str, default='../output',
                    help='just need output folder, not filename. if None, result will not save.')
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    default=False, help='draw keypoints numbers when rendering results')
    ap.add_argument('--debug_track', action='store_true',
                    default=True, help='visualize tracker algorithm with bboxes')
    return ap.parse_args()

def get_suffix(args, cfg):
    suffix = []
    suffix.append(cfg.POSE.name)
    if args.task != 'pose':
        suffix.extend([cfg.TRACKER.name, cfg.TRACKER.dataset_name, cfg.TRACKER.reid_name])
        if args.task == 'action':
            suffix.extend([cfg.CLASSIFIER.name, 'torch'])
    return suffix

def main():
     # Configs
    args = get_args()
    cfg = Config(args.config)
    pose_kwargs = cfg.POSE
    clf_kwargs = cfg.CLASSIFIER
    tracker_kwargs = cfg.TRACKER

    # Initiate video/webcam
    source = args.source if args.source else 0
    video = Video(source)

    ## Initiate trtpose, deepsort and action classifier
    pose_estimator = get_pose_estimator(**pose_kwargs)
    if args.task != 'pose':
        tracker = get_tracker(**tracker_kwargs)
        if args.task == 'action':
            action_classifier = get_classifier(**clf_kwargs)

    ## initiate drawer and text for visualization
    drawer = Drawer(draw_numbers=args.draw_kp_numbers)
    user_text = {
        'text_color': 'green',
        'add_blank': True,
        'Mode': args.task,
        # MaxDist: cfg.TRACKER.max_dist,
        # MaxIoU: cfg.TRACKER.max_iou_distance,
    }

    # loop over the video frames
    for bgr_frame in video:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # predict pose estimation
        start_pose = time.time()
        predictions = pose_estimator.predict(rgb_frame, get_bbox=True) # return predictions which include keypoints in trtpose order, bboxes (x,y,w,h)
        # if no keypoints, update tracker's memory and it's age
        if len(predictions) == 0 and args.task != 'pose':
            debug_img = bgr_frame
            tracker.increment_ages()
        else:
            # draw keypoints only if task is 'pose'
            if args.task != 'pose':
                # Tracking
                # start_track = time.time()
                predictions = utils.convert_to_openpose_skeletons(predictions)
                predictions, debug_img = tracker.predict(rgb_frame, predictions,
                                                                debug=args.debug_track)
                # end_track = time.time() - start_track

                # Action Recognition
                if len(predictions) > 0 and args.task == 'action':
                    predictions = action_classifier.classify(predictions)

        end_pipeline = time.time() - start_pose
        # add user's desired text on render image
        user_text.update({
            'Frame': video.frame_cnt,
            'Speed': '{:.1f}ms'.format(end_pipeline*1000),
        })

        # draw predicted results on bgr_img with frame info
        render_image = drawer.render_frame(bgr_frame, predictions, **user_text)

        if video.frame_cnt == 1 and args.save_folder:
            # initiate writer for saving rendered video.
            output_suffix = get_suffix(args, cfg)
            output_path = video.get_output_file_path(
                args.save_folder, suffix=output_suffix)
            writer = video.get_writer(render_image, output_path, fps=30)

            if args.debug_track and args.task != 'pose':
                debug_output_path = output_path[:-4] + '_debug.avi'
                debug_writer = video.get_writer(debug_img, debug_output_path)
            print(f'[INFO] Saving video to : {output_path}')
        # show frames
        try:
            if args.debug_track and args.task != 'pose':
                debug_writer.write(debug_img)
                utils.show(debug_img, window='debug_tracking')
            if args.save_folder:
                writer.write(render_image)
            utils.show(render_image, window='webcam' if isinstance(source, int) else osp.basename(source))
        except StopIteration:
            break
    if args.debug_track and args.task != 'pose':
        debug_writer.release()
    if args.save_folder and len(predictions) > 0:
        writer.release()
    video.stop()

if __name__ == '__main__':
    main()
