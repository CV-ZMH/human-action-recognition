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
from utils import utils, drawer


def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # configs
    ap.add_argument('--task', choices=['pose', 'track', 'action'], default='action',
                    help='inference task for pose estimation, action recognition or tracking')

    ap.add_argument("--config", type=str,
                    default="../configs/infer_trtpose_deepsort_dnn.yaml",
                    help='all inference configs for full action recognition pipeline.')
    # inference source
    ap.add_argument('--source',
                    default='../test_data/aung_la.mp4',
                    help='input source for pose estimation, if None, it wiil use webcam by default')
    # save path and visualization info
    ap.add_argument('--save_folder', type=str, default='../output',
                    help='just need output folder, not filename. if None, result will not save.')
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    help='draw keypoints numbers of each person')
    ap.add_argument('--debug_track', action='store_true',
                    # default=True,
                    help='show and save for tracking bbox and detection bbox')

    return ap.parse_args()

def get_suffix(args, cfg):
    suffix = []
    suffix.append(cfg.POSE.name)
    if args.task != 'pose':
        suffix.extend([cfg.TRACKER.name, cfg.TRACKER.reid_net])
        if args.task == 'action':
            suffix.extend([cfg.CLASSIFIER.name])
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
    try:
        # loop over the video frames
        for bgr_frame in video:
            rgb_frame= cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            # predict pose estimation
            start_pose = time.time()
            keypoints_list = pose_estimator.predict(rgb_frame)
            end_pose = time.time() - start_pose

            # if no keypoints, update tracked's time_since_update and it's age
            if len(keypoints_list) == 0 and args.task != 'pose':
                debug_img = bgr_frame
                tracker.increment_ages()
            else:
                # draw keypoints only if task is 'pose'
                if args.task == 'pose':
                    drawer.draw_persons_keypoints(
                        bgr_frame,
                        keypoints_list,
                        draw_numbers=args.draw_kp_numbers
                    )
                # tracking / actions
                else:
                    # preprocess trtpose keypoints for tracking and action classifier
                    skeletons_list = utils.convert_to_skeletons(keypoints_list)
                    bboxes = utils.keypoints_to_bbox(keypoints_list, rgb_frame)

                    # pass keypoints' bboxes to tracker
                    start_track = time.time()
                    tracks, debug_img = tracker.update(bboxes, rgb_frame, debug=args.debug_track)
                    end_track = time.time() - start_track

                    # classify tracked skeletons' actions
                    if len(tracks) > 0:
                        if args.task == 'action':
                            tracked_keypoints = {
                                track["track_id"]: skeletons_list[track["detection_index"]]
                                for track in tracks
                                }
                            actions = action_classifier.classify(tracked_keypoints)

                        # draw keypoints, tracks, actions on current frame
                        drawer.draw_frame(
                            bgr_frame,
                            tracks,
                            keypoints_list,
                            draw_numbers=args.draw_kp_numbers,
                            actions=actions if args.task=='action' else None
                        )

            end_pipeline = time.time() - start_pose
            # add desired text on current frame with key value pairs
            final_img = drawer.draw_frame_info(
                bgr_frame,
                color='green',
                mode=args.task,
                FPS=f'{video.fps:.3f}',
                Frame=f'{video.frame_cnt}/{video.total_frames}',
                MaxDist=cfg.TRACKER.max_dist,
                MaxIoU=cfg.TRACKER.max_iou_distance,
                # TotalSpeed='{:.3f}s'.format(end_pipeline),
                # TotalTracks=len(tracks) if len(keypoints_list) > 0 else 0,
                # TrackSpeed='{:.3f}s'.format(end_track) if len(keypoints_list)>0 else 0,
                )

            if video.frame_cnt == 1 and args.save_folder:
                output_suffix = get_suffix(args, cfg)
                output_path = video.get_output_file_path(
                    args.save_folder,
                    suffix=output_suffix
                    )
                writer = video.get_writer(final_img, output_path)
                if args.debug_track and args.task != 'pose':
                    debug_output_path = output_path[:-4] + '_debug.avi'
                    debug_writer = video.get_writer(debug_img, debug_output_path)

                print(f'[INFO] Saving video to : {output_path}')

            if args.debug_track and args.task != 'pose':
                debug_writer.write(debug_img)
                key = video.show(debug_img, winname='debug_tracking')

            if args.save_folder:
                writer.write(final_img)

            key = video.show(
                final_img,
                winname='webcam' if isinstance(source, int) else osp.basename(source)
            )

            if key == ord('q') or key == 27:
                break

        if args.debug_track and args.task != 'pose':
            debug_writer.release()
        if args.save_folder and len(keypoints_list) > 0:
            writer.release()

    finally:
        video.stop()


if __name__ == '__main__':
    main()
