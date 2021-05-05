# -*- coding: utf-8 -*-
import cv2
import numpy as np


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

l_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), #  head
           (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
           (17, 11), (17, 12), # body
           (11, 13), (12, 14), (13, 15), (14, 16)]

p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102),
           (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255),
           (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255),
           (77, 255, 191), (127, 77, 255), (77, 255, 127),
           (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
            (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
            (77, 222, 255), (255, 156, 127),
            (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]
    return color

def vis_frame(image, outputs, all_keypoints, args):
    """Draw keypoint tracking result on image"""
    height, width = image.shape[:2]
    for track_id, track in outputs.items():
        color = get_color_fast(track_id) if args.tracking else BLUE

        thickness = 2 if image.shape[1]/640 > 1.3 else 1
        visibilities = []
        points = {}
        # draw keypoints and keypoint numbers on image
        for kp in all_keypoints[track['kp_index']]:
            if kp[1]==0 or kp[2]==0: # check missing keypoints
                visibilities.append(kp[0])
                continue
            center = int(kp[1] * width + 0.5) , int(kp[2] * height + 0.5)
            points[kp[0]] = center
            cv2.circle(image, center, thickness, p_color[int(kp[0])], thickness+2)
            if args.draw_kp_numbers:
                cv2.putText(image, str(int(kp[0])), center, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.6, (0,0,255), thickness)

         # draw keypoints connection on image
        for pair_order, pair in enumerate(l_pairs):
            if pair[0] in visibilities or pair[1] in visibilities: continue
            if args.tracking:
                cv2.line(image, points[pair[0]], points[pair[1]], color, thickness)
            else:
                cv2.line(image, points[pair[0]], points[pair[1]], line_color[pair_order], thickness)

        # draw track bbox and track_id
        if args.tracking:
            x1, y1, x2, y2 = map(int, track['bbox'])
            label = '{}{:d}'.format("", track_id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(
                image, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(image, label, (x1, y1 + t_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 2, BLACK, 2)

# =============================================================================
#     if args.add_feature_template:
#         x1, y1, x2, y2 = outputs[keypoints[0]]
#         template = img_disp[y1:y2, x1:x2]
#         template = cv2.copyMakeBorder(template, 5, 5, 5, 5,
#                                       cv2.BORDER_CONSTANT)
#         #add features in top right corner
#         img_disp[:template.shape[0], - template.shape[1]:] = template
# =============================================================================

def drawFrameInfo(img, **kwargs):
    # draw frame info
    y0, dy = 20, 20
    texts = [f'{k} : {v}' for k,v in kwargs.items()]
    for i, line in enumerate(texts):
        y = y0 + i*dy
        cv2.putText(img, line, (5, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, BLACK, 2)

# =============================================================================
# #%% debug functions
# def compute_color_for_labels(label):
#     """
#     Simple function that adds fixed color depending on the class
#     """
#     palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
#     color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
#     return tuple(color)
# def draw_detection(img, bboxes):
#     for idx, bbox in enumerate(bboxes):
#         x1 = int(bbox[0] - (bbox[2]/2))
#         x2 = int(bbox[0] + (bbox[2]/2))
#         y1 = int(bbox[1] - (bbox[3]/2))
#         y2 = int(bbox[1] + (bbox[3]/2))
#         # print(x1,y1,x2,y2)
#         label = '{}{:d}'.format("", idx)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), BLACK, 2)
#         cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), BLACK, -1)
#         cv2.putText(img, label, (x1, y1 + t_size[1] + 4),
#                     cv2.FONT_HERSHEY_PLAIN, 2, WHITE, 2)
#
# =============================================================================

