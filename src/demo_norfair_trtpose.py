import os
import time
import cv2
import argparse
import torch
import numpy as np

from lib.tracker import norfair
from lib.utils import utils, vis, parser
from lib.pose_estimation.trtpose import TrtPose
from lib.tracker.norfair import Detection, Tracker
from lib.tracker.deepsort.get_extractor import Extractor

frame_skip_period = 1
detection_threshold = 0.01
distance_threshold = 0.5

infer_yaml = '../configs/inference_all.yaml'
webcam = False
input_source = '/home/zmh/hdd/Test_Videos/Tracking/fun_theory_2.mp4'
total_joints = 8
leg_joints = 4
save_folder = '../output'
mode = 'track'



def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


# def get_metric(matching_threshold, budget=None):
#         matching_threshold = matching_threshold
#         budget = budget
#         samples = {}



def keypoints_distance(detected_pose, tracked_pose):
    distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
    match_num = np.count_nonzero(
        (distances < keypoint_dist_threshold)
        * (detected_pose.scores > detection_threshold)
        * (tracked_pose.last_detection.scores > detection_threshold)
    )
    return 1 / (1 + match_num)

def to_norfair_detection(trtpose_keypoints, img):
    new_points = trtpose_keypoints[..., 1:].copy()
    if new_points.any():
    # new_points = np.empty((trtpose_keypoints.shape[0], 18, 3))
        new_points[..., 0] = new_points[..., 0] * img.shape[1]
        new_points[..., 1] = new_points[..., 1] * img.shape[0]
    # new_points[..., 2] = 0.01
    return new_points

def get_keypoints(humans, counts, peaks):
    """Get all persons keypoint"""

    all_keypoints = np.zeros((counts, 18, 4), dtype=np.float64) #  counts contain num_persons
    for count in range(counts):
        human = humans[0][count]
        C = human.shape[0]
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]
                peak = (j, float(peak[1]), float(peak[0]), 0.9)
                all_keypoints[count, j] = peak
            else:
                peak = (j, 0., 0., 0.01)
                all_keypoints[count, j] = peak

    return all_keypoints


def get_features(extractor, trtpose_keypoints, img):
    height, width = img.shape[:2]
    openpose_keypoints = utils.trtpose_to_openpose(trtpose_keypoints)
    bbox_xywh = utils.get_skeletons_bboxes(openpose_keypoints, img_rgb)
    im_crops = []
    def _xywh_to_xyxy(xywh):
        x, y, w, h = xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2

    for box in bbox_xywh:
        x1, y1, x2, y2 = _xywh_to_xyxy(box)
        im = img[y1:y2, x1:x2]
        im_crops.append(im)
    if im_crops:
        features = extractor(im_crops)
        if len(features.shape)==1:
            features = np.expand_dims(features,0)
    else:
        features = np.array([])
    return features


# Configs

cfg = parser.YamlParser(config_file=infer_yaml)

# Initiate video/webcam
cap = cv2.VideoCapture(0 if webcam else input_source)
assert cap.isOpened(),  f"Can't open video : {input_source}"
display_name = 'webcam' if webcam else os.path.basename(input_source)

# pose estimator
cfg.POSE.trtpose.weight = os.path.join(*cfg.POSE.trtpose.weight)
trtpose = TrtPose(**cfg.POSE.trtpose)

# tracker
tracker = Tracker(
        distance_function=keypoints_distance,
        distance_threshold=distance_threshold,
        detection_threshold=detection_threshold,
        point_transience=2,
    )

# reid feature extractor
extractor = Extractor(**cfg.TRACKER.deepsort)


keypoint_dist_threshold = 256 / 25
frame_cnt = 0
try:
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 640, 480)

    while True:
        ret, img_bgr = cap.read()
        if not ret: break
        img_disp = img_bgr.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # predict keypoints
        if frame_cnt % frame_skip_period == 0:
            start_pose = time.time()
            counts, objects, peaks = trtpose.predict(img_rgb)
            trtpose_keypoints = get_keypoints(objects, counts, peaks)
            trtpose_keypoints = trtpose.remove_persons_with_few_joints(
                trtpose_keypoints,
                min_total_joints=total_joints, #TODO change parameter names
                min_leg_joints=leg_joints,
                include_head=True
                )
            # openpose_keypoints = utils.trtpose_to_openpose(trtpose_keypoints)
            # bboxes = utils.get_skeletons_bboxes(openpose_keypoints, img_bgr)

            features = get_features(extractor, trtpose_keypoints, img_rgb)
            new_points = to_norfair_detection(trtpose_keypoints, img_disp)
            detections = (
                [] if not new_points.any()
                else [Detection(p, scores=s, data=f) for (p, s, f) \
                      in zip(new_points[..., :2], new_points[..., 2], features)]
            )
            tracked_objects = tracker.update(
                detections=detections, period=frame_skip_period
            )
            # norfair.draw_points(img_disp, detections)
        else:
            tracked_objects = tracker.update()

        norfair.draw_tracked_objects(img_disp, tracked_objects, id_thickness=3, radius=2)
        # print(tracked_objects)

        if frame_cnt == 0 and save_folder:
            os.makedirs(save_folder, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save_name = os.path.join(
                            save_folder, '{}_trtpose_{}_{}_{}.avi'.format(
                                os.path.basename(input_source[:-4]) if not webcam else 'webcam',
                                'norfair',
                                mode, cfg.POSE.trtpose.size
                            ))

            writer = cv2.VideoWriter(
                save_name, fourcc, 20.0,
                (img_disp.shape[1], img_disp.shape[0])
                )
        frame_cnt += 1
        writer.write(img_disp)
        cv2.imshow(display_name, img_disp)
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break

finally:
    if save_folder:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()