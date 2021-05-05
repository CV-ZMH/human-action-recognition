# -*- coding: utf-8 -*-
import sys
import os
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

import cv2
import numpy as np
import argparse

from pose2d.pose import Pose
from utils import lib_images_io, lib_plot, lib_commons
from utils.lib_tracker import Tracker
from utils.lib_classifier import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test action recognition on \n"
        "(1) a video, (2) a folder of images, (3) or web camera.")
    parser.add_argument("-m", "--model_path", required=False,
                        default='../model/trained_classifier.pickle')
    parser.add_argument("-t", "--data_type", required=False, default='video',
                        choices=["video", "folder", "webcam"])
    parser.add_argument("-p", "--data_path", required=False, default="/home/zmh/hdd/Test_Videos/Tracking/jordyn_dance.mp4",
                        help="path to a video file, or images folder, or webcam. \n"
                        "For video and folder, the path should be "
                        "absolute or relative to this project's root. "
                        "For webcam, either input an index or device name. ")
    parser.add_argument("-o", "--output_folder", required=False, default='../output/',
                        help="Which folder to save result to.")

    return parser.parse_args()

def get_dst_folder_name(src_data_type, src_data_path):
    ''' Compute a output folder name based on data_type and data_path.
        The final output of this script looks like this:
            DST_FOLDER/folder_name/vidoe.avi
            DST_FOLDER/folder_name/skeletons/XXXXX.txt
    '''

    assert(src_data_type in ["video", "folder", "webcam"])

    if src_data_type == "video":  # /root/data/video.avi --> video
        folder_name = os.path.basename(src_data_path).split(".")[-2]

    elif src_data_type == "folder":  # /root/data/video/ --> video
        folder_name = src_data_path.rstrip("/").split("/")[-1]

    elif src_data_type == "webcam":
        # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
        folder_name = lib_commons.get_time_string()

    return folder_name

def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            src_data_path,
            sample_interval=src_video_sample_interval)

    elif src_data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=src_data_path)

    elif src_data_type == "webcam":
        if src_data_path == "":
            webcam_idx = 0
        elif src_data_path.isdigit():
            webcam_idx = int(src_data_path)
        else:
            webcam_idx = src_data_path
        images_loader = lib_images_io.ReadFromWebcam(
            src_webcam_max_fps, webcam_idx)
    return images_loader

class MultiPersonClassifier(object):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, window_size, human_id)

    def classify(self, dict_id2skeleton):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)  # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        ''' Get the classifier based on the person id.
        Arguments:
            id {int or "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]

def draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                    pose, multiperson_classifier):
    ''' Draw skeletons, labels, and prediction scores onto image for display '''

    # Resize to a proper size for display
    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
    img_disp = cv2.resize(img_disp,
                          dsize=(desired_cols, img_disp_desired_rows))

    # Draw all people's skeleton
    pose.draw2D(img_disp, humans, draw_numbers=True)

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # scale the y data back to original
            skeleton[1::2] = skeleton[1::2] / scale_h
            # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            lib_plot.draw_action_result(img_disp, id, skeleton, label)

    # Add blank to the left for displaying prediction scores of each class
    img_disp = lib_plot.add_white_region_to_left_of_image(img_disp)

    cv2.putText(img_disp, "Frame:" + str(ith_img),
                (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 0), thickness=2)

    # Draw predicting score for only 1 person
    if len(dict_id2skeleton):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')
        classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp

def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
    '''
    In each image, for each skeleton, save the:
        human_id, label, and the skeleton positions of length 18*2.
    So the total length per row is 2+36=38
    '''
    skels_to_save = []
    for human_id in dict_id2skeleton.keys():
        label = dict_id2label[human_id]
        skeleton = dict_id2skeleton[human_id]
        skels_to_save.append([[human_id, label] + skeleton.tolist()])
    return skels_to_save


# -- Main
if __name__ == "__main__":

    args = parse_args()
    src_data_type = args.data_type
    src_data_path = args.data_path
    src_model_path = args.model_path

    dst_folder_name = get_dst_folder_name(src_data_type, src_data_path)

    # -- settings
    cfg_all = lib_commons.read_yaml(os.path.join(ROOT, 'config', 'config.yaml'))
    cfg = cfg_all[os.path.basename(__file__)]

    classes = np.array(cfg_all['classes'])
    skeleton_filename_format = cfg_all['skeleton_filename_format']

    # action recogn
    window_size = int(cfg_all['features']['window_size'])

    # output folder
    dst_folder = os.path.join(args.output_folder, dst_folder_name)
    dst_skeleton_folder_name = cfg['output']['skeleton_folder_name']
    dst_video_name = cfg['output']['video_name']

    # framerate of output video.avi
    dst_video_fps = float(cfg['output']['video_fps'])
    src_webcam_max_fps = float(cfg['settings']['source']['webcam_max_framerate'])
    src_video_sample_interval = int(cfg['settings']['source']['video_sample_interval'])

    # trtpose setting
    cfg_model = cfg_all['trtpose']
    img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])
    # -- Detector, tracker, classifier
    pose = Pose(cmap_threshold=0.1, link_threshold=0.1, **cfg_model)

    multiperson_tracker = Tracker()
    multiperson_classifier = MultiPersonClassifier(src_model_path, classes)

    # -- Image reader and displayer
    images_loader = select_images_loader(src_data_type, src_data_path)

    # -- Init output
    # output folder
    os.makedirs(dst_folder, exist_ok=True)
    os.makedirs(os.path.join(dst_folder, dst_folder_name), exist_ok=True)

    # video writer
    video_writer = lib_images_io.VideoWriter(
        os.path.join(dst_folder, dst_video_name), dst_video_fps)

    cv2.namedWindow(src_data_type, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(src_data_type, 640, 480)
    # -- Read images and process
    try:
        ith_img = -1
        while images_loader.has_image():
            # -- Read image
            img_bgr = images_loader.read_image()
            img_disp = img_bgr.copy()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ith_img += 1
            # -- Detect skeletons
            humans = pose.predict(img_rgb)
            humans = pose.pose.remove_persons_with_few_joints(humans,
                                                            min_total_joints=6,
                                                            min_leg_joints=1,
                                                            include_head=False)
            skeletons, scale_h = pose.keypoints_to_skels_list(humans)
            # -- Track people
            dict_id2skeleton = multiperson_tracker.track(
                skeletons)  # int id -> np.array() skeleton

            # -- Recognize action of each person
            if len(dict_id2skeleton):
                dict_id2label = multiperson_classifier.classify(
                    dict_id2skeleton)

            # -- Draw
            img_disp = draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                                       pose, multiperson_classifier)

            # Print label of a person
            if len(dict_id2skeleton):
                min_id = min(dict_id2skeleton.keys())
                # print("prediced label is :", dict_id2label[min_id])

            # -- Display image, and write to video.avi
            cv2.imshow(src_data_type, img_disp)
            key = cv2.waitKey(1)
            if key==27 or key==ord('q'):
                break

            video_writer.write(img_disp)
            # -- Get skeleton data and save to file
            skels_to_save = get_the_skeleton_data_to_save_to_disk(
                dict_id2skeleton)
            lib_commons.save_listlist(
                os.path.join(dst_folder, dst_skeleton_folder_name +
                skeleton_filename_format.format(ith_img)),
                skels_to_save)
    finally:
        cv2.destroyAllWindows()
        video_writer.stop()
        print("Program ends")
        sys.exit()
