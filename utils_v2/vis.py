# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from .commons import *

def draw_frame_info(img, **kwargs):
    # draw frame info
    y0, dy = 20, 20
    texts = [f'{k} : {v}' for k,v in kwargs.items()]
    for i, line in enumerate(texts):
        y = y0 + i*dy
        cv2.putText(img, line, (5, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, BLACK, 2)

def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]
    return color

def draw_action_recognition(image,
                            tracks,
                            all_keypoints,
                            actions):
    """Draw skeleton pose, tracking id and action result on image"""

    height, width = image.shape[:2]
    thickness = 2 if height*width > (720*960) else 1

    for track_id, track in tracks.items():
        color = get_color_fast(track_id)

        # draw keypoints
        points = {}
        visibilities = []
        for keypoints in all_keypoints[track['kp_index']]:
            if keypoints[1] == 0 or keypoints[2] == 0:
                visibilities.append(keypoints[0])
                continue
            center = int(keypoints[1] * width + 0.5), int(keypoints[2] * height + 0.5)
            points[keypoints[0]] = center
            cv2.circle(image, center, thickness, p_color[int(keypoints[0])], thickness+2)

        # draw keypoints connection
        for pair in l_pairs:
            if pair[0] in visibilities or pair[1] in visibilities:
                continue
            cv2.line(image, points[pair[0]], points[pair[1]], color, thickness)

        # draw track bbox
        x1, y1, x2, y2 = map(int, track['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # draw text over rectangle background
        label = '{:d}: {}'.format(track_id, actions[track_id])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, thickness)[0]
        yy = (y1 - t_size[1] - 5, y1 - t_size[1] + 14) if y1 - t_size[1] - 5 > 0 \
            else (y1 + t_size[1] + 5, y1 + t_size[1])

        cv2.rectangle(image, (x1, y1), (x1 + t_size[0]+1, yy[0]), color, -1)
        cv2.putText(image, label, (x1, yy[1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, BLACK, thickness)

# =============================================================================
# def draw_keypoint_tracking(image, outputs, all_keypoints, args):
#     """Draw keypoint tracking result on image"""
#
#     height, width = image.shape[:2]
#     thickness = 2 if image.shape[1]/640 > 1.3 else 1
#     for track_id, track in outputs.items():
#         color = get_color_fast(track_id) if args.tracking else BLUE
#
#         visibilities = []
#         points = {}
#         # draw keypoints and keypoint numbers on image
#         for kp in all_keypoints[track['kp_index']]:
#             if kp[1]==0 or kp[2]==0: # check missing keypoints
#                 visibilities.append(kp[0])
#                 continue
#             center = int(kp[1] * width + 0.5) , int(kp[2] * height + 0.5)
#             points[kp[0]] = center
#             cv2.circle(image, center, thickness, p_color[int(kp[0])], thickness+2)
#             if args.draw_kp_numbers:
#                 cv2.putText(image, str(int(kp[0])), center, cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                         0.6, (0,0,255), thickness)
#
#           # draw keypoints connection on image
#         for pair_order, pair in enumerate(l_pairs):
#             if pair[0] in visibilities or pair[1] in visibilities: continue
#             if args.tracking:
#                 cv2.line(image, points[pair[0]], points[pair[1]], color, thickness)
#             else:
#                 cv2.line(image, points[pair[0]], points[pair[1]], line_color[pair_order], thickness)
#
#         # draw track bbox and track_id
#         if args.tracking:
#             x1, y1, x2, y2 = map(int, track['bbox'])
#             label = '{}{:d}'.format("", track_id)
#             t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#             cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
#             cv2.rectangle(
#                 image, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#             cv2.putText(image, label, (x1, y1 + t_size[1] + 4),
#                         cv2.FONT_HERSHEY_PLAIN, 2, BLACK, 2)
#
#     if args.add_feature_template:
#         x1, y1, x2, y2 = outputs[keypoints[0]]
#         template = img_disp[y1:y2, x1:x2]
#         template = cv2.copyMakeBorder(template, 5, 5, 5, 5,
#                                       cv2.BORDER_CONSTANT)
#         #add features in top right corner
#         img_disp[:template.shape[0], - template.shape[1]:] = template
# =============================================================================

#%% Draw Fuction for Trtpose
JointPairs = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 17), # head
              (17, 5), (17, 6), (5, 7), (6, 8), (7, 9), (8, 10), # arms
              (17, 11), (17, 12), (11, 13), (12, 14), (13, 15), (14, 16), # legs
              (3, 5), (4, 6) # ear to arms
              ]
JointPairsRender = JointPairs[:-2]
JointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def draw_trtpose(image, all_keypoints, draw_circle=False, draw_numbers=False, skip_from=-1):
    """Draw trtpose persons' keypoints"""
    thickness = 2 if image.shape[1]/640 > 1.3 else 1
    for keypoints in all_keypoints:
        visibilities = []
        centers = {}

        # draw points on image
        for kp in keypoints:
            if kp[1]==0 or kp[2]==0:
                visibilities.append(kp[0])
                continue
            center = int(kp[1] * image.shape[1] + 0.5) , int(kp[2] * image.shape[0] + 0.5)
            centers[kp[0]] = center
            if draw_circle:
                cv2.circle(image, center, thickness, BLACK, thickness+2)
            if draw_numbers:
                cv2.putText(image, str(int(kp[0])), center, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.6 if thickness==1 else 1, RED, 1)

         # draw line on image
        for pair_order, pair in enumerate(JointPairsRender):
            if pair[0] < skip_from or pair[1] < skip_from: continue
            if pair[0] in visibilities or pair[1] in visibilities: continue
            cv2.line(image, centers[pair[0]], centers[pair[1]], JointColors[pair_order], 2)

#%% Plot confusion matric for training
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          size=None):
    """ (Copied from sklearn website)
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Display normalized confusion matrix ...")
    else:
        print('Display confusion matrix without normalization ...')

    fig, ax = plt.subplots()
    if size is None:
        size = (12, 8)
    fig.set_size_inches(size[0], size[1])

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim([-0.5, len(classes)-0.5])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm