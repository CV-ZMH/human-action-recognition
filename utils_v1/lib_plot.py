

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


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

    # print(cm)

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


# Drawings ==============================================================

def draw_action_result(img_display, id, skeleton, action_label=''):
    font = cv2.FONT_HERSHEY_COMPLEX

    x_points = skeleton[skeleton>0][0::2]
    y_points = skeleton[skeleton>0][1::2]
    minX, maxX = x_points.min(), x_points.max()
    minY, maxY = y_points.min(), y_points.max()

    minx = int(minX * img_display.shape[1])
    miny = int(minY * img_display.shape[0])
    maxx = int(maxX * img_display.shape[1])
    maxy = int(maxY * img_display.shape[0])

    # Draw bounding box
    # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
    cv2.rectangle(img_display, (minx, miny), (maxx, maxy), (0, 255, 0), 2)

    # Draw text at left corner
    box_scale = max(
        0.5, min(2.0, (1.0*(maxx - minx)/img_display.shape[1] / (0.3))**(0.5)))

    TEST_COL = int(minx + 5 * box_scale)
    TEST_ROW = int(miny - 10 * box_scale)
    # TEST_ROW = int( miny)
    # TEST_ROW = int( skeleton[3] * img_display.shape[0])

    cv2.putText(img_display, str(id % 10) +''+ action_label, (TEST_COL, TEST_ROW),
                              font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


def add_blank_region_to_left_of_image(img_disp, color='black'):
    r, c, d = img_disp.shape
    region = np.zeros((r, int(c/4), d), np.uint8)
    if color=='white':
        region += 255

    img_disp = np.hstack((region, img_disp))
    return img_disp
