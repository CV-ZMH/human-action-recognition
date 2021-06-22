# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from .commons import *

def draw_frame_info(img, color='red', **kwargs):
    """Draw texts as provided kwargs in the left corner of the frame"""

    # add blank area in left side to display the texts
    h, w, d = img.shape
    blank = np.zeros((h, 200, d), dtype=np.uint8)
    img_disp = np.hstack((blank, img))
    # draw texts
    color = colors[color.lower()]
    texts = [f'{k}: {v}' for k,v in kwargs.items()]
    y0, dy = 25, 50
    for i, line in enumerate(texts):
        y = y0 + i*dy
        cv2.putText(img_disp, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img_disp

def get_color_fast(idx):
    color_pool = list(colors.values())[:-1] # no use black color for tracking bbox
    color = color_pool[idx % len(color_pool)]
    return color

def draw_frame(image, tracks, all_keypoints, actions=None, **kwargs):
    """Draw skeleton pose, tracking id and action result on image.
    Check kwargs in func: `draw_trtpose`
    """

    height, width = image.shape[:2]
    thickness = 2 if height*width > (720*960) else 1

    # Draw each of the tracked skeletons and actions text
    for track in tracks:
        track_id = track['track_id']
        color = get_color_fast(track_id)

        # draw keypoints
        keypoints = all_keypoints[track['detection_index']]
        draw_trtpose(image,
                     keypoints,
                     thickness=thickness,
                     line_color=color,
                     **kwargs)

        # draw track bbox
        x1, y1, x2, y2 = map(int, track['track_bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # draw text over rectangle background
        label = actions.get(track_id, '') if actions else ''
        label = '{:d}: {}'.format(track_id, label)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, thickness)[0]
        yy = (y1 - t_size[1] - 6, y1 - t_size[1] + 14) if y1 - t_size[1] - 5 > 0 \
            else (y1 + t_size[1] + 6, y1 + t_size[1])

        cv2.rectangle(image, (x1, y1), (x1 + t_size[0]+1, yy[0]), color, -1)
        cv2.putText(image, label, (x1, yy[1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, colors['black'], thickness)

def draw_persons_keypoints(image, all_keypoints, **kwargs):
    """Draw all persons' keypoints.
    Check kwargs in func: `draw_trtpose`
    """
    height, width = image.shape[:2]
    thickness = 2 if height*width > (720*960) else 1
    for keypoints in all_keypoints:
        draw_trtpose(image, keypoints, thickness=thickness, **kwargs)

def draw_trtpose(image,
                 keypoints,
                 thickness=2,
                 draw_points=True,
                 draw_numbers=False,
                 skip_from=-1,
                 line_color=None):
    """Draw keypoints and their connections as trtpose format"""

    visibilities = []
    centers = {}
    # draw points on image
    for kp in keypoints:
        if kp[1]==0 or kp[2]==0:
            visibilities.append(kp[0])
            continue
        center = int(kp[1] * image.shape[1] + 0.5) , int(kp[2] * image.shape[0] + 0.5)
        centers[kp[0]] = center
        if draw_points:
            cv2.circle(image, center, thickness, points_color[int(kp[0])], thickness+2)
        if draw_numbers:
            cv2.putText(image, str(int(kp[0])), center, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.6 if thickness==1 else 1, colors['red'], 1)

     # draw line on image
    for pair_idx, pair in enumerate(limb_pairs):
        if pair[0] < skip_from or pair[1] < skip_from: continue
        if pair[0] in visibilities or pair[1] in visibilities: continue
        if line_color:
            cv2.line(image, centers[pair[0]], centers[pair[1]], line_color, thickness)
        else:
            # print(pair_idx, LR[pair_idx])
            cv2.line(image, centers[pair[0]], centers[pair[1]], \
                     colors['blue'] if LR[pair_idx] else colors['red'],
                     thickness)

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