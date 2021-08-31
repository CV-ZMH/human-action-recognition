import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from utils.commons import *


class Drawer:
    def __init__(self, draw_points=True, draw_numbers=False, color='green', thickness=1):
        self.draw_points = draw_points
        self.draw_numbers = draw_numbers
        self.color = COLORS[color]
        self.scale = 0.6 if thickness <= 2 else 0.8
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_COMPLEX

    def render_frame(self, image, predictions, **user_text_kwargs):
        """Draw all persons [skeletons / tracked_id / action] annotations on image
        in trtpose keypoint format.
        """
        render_frame = image.copy()
        def _scale_keypoints(pred):
            if pred.keypoints[..., 1:].max() <= 1:
                pred.keypoints[..., 1:] *= render_frame.shape[:2][::-1]
            pred.keypoints = pred.keypoints.astype(np.int16)
            return pred

        predictions = [_scale_keypoints(pred) for pred in predictions]
        # draw the results
        for pred in predictions:
            if pred.color is not None: self.color = pred.color
            self.draw_trtpose(render_frame, pred)
            if pred.bbox is not None:
                self.draw_bbox_label(render_frame, pred)

        if len(user_text_kwargs)>0:
            render_frame = self.add_user_text(render_frame, **user_text_kwargs)
        return render_frame

    def draw_trtpose(self, image, pred):
        """Draw skeletons on image with trtpose keypoint format"""

        visibilities = []
        # draw circle and keypoint numbers
        for kp in pred.keypoints:
            if kp[1]==0 or kp[2]==0:
                visibilities.append(kp[0])
                continue
            if self.draw_points:
                cv2.circle(image, (kp[1],kp[2]), self.thickness, self.color, self.thickness+2)
            if self.draw_numbers:
                cv2.putText(image, str(kp[0]), (kp[1],kp[2]), self.font,
                            self.scale - 0.2, COLORS['blue'], self.thickness)

        # draw skeleton connections
        for pair in LIMB_PAIRS:
            if pair[0] in visibilities or pair[1] in visibilities: continue
            start, end = map(tuple, [pred.keypoints[pair[0]], pred.keypoints[pair[1]]])
            cv2.line(image, start[1:], end[1:], self.color, self.thickness)

    def draw_bbox_label(self, image, pred):
        scale = self.scale - 0.1
        x1, y1, x2, y2 = pred.bbox.astype(np.int16)
        # draw person bbox
        cv2.rectangle(image, (x1,y1), (x2,y2), self.color, self.thickness)

        def get_label_position(label, is_track=False):
            w, h = cv2.getTextSize(label, self.font, scale, self.thickness)[0]
            offset_w, offset_h = w + 3, h + 5
            xmax = x1 + offset_w
            is_upper_pos = True
            if (y1 - offset_h) < 0 or is_track:
                ymax = y1 + offset_h
                y_text = ymax - 2
            else:
                ymax = y1 - offset_h
                y_text = y1 - 2
                is_upper_pos = False
            return xmax, ymax, y_text, is_upper_pos

        if pred.id:
            track_label = f'{pred.id}'
            *track_loc, is_upper_pos = get_label_position(track_label, is_track=True)
            cv2.rectangle(image, (x1, y1), (track_loc[0], track_loc[1]), self.color, -1)
            cv2.putText(image, track_label, (x1+1, track_loc[2]), self.font,
                        scale, COLORS['black'], self.thickness)

            # draw text over rectangle background
            if pred.action[0]:
                action_label = '{}: {:.2f}'.format(*pred.action) if pred.action[0] else ''
                if not is_upper_pos:
                    action_label = f'{track_label}-{action_label}'
                action_loc = get_label_position(action_label)
                cv2.rectangle(image, (x1, y1), (action_loc[0], action_loc[1]), self.color, -1)
                cv2.putText(image, action_label, (x1+1, action_loc[2]), self.font,
                            scale, COLORS['black'], self.thickness)


    def add_user_text(self, image, text_color='red', add_blank=True, **user_text):
        h, w, d = image.shape
        if add_blank:
            size = (h, 200, d) if h > w/1.5 else (200, w, d)
            blank = np.zeros(size, dtype=np.uint8)
            image = np.hstack((blank, image)) if h > w/1.5 else np.vstack((blank, image))
        # draw texts
        if len(user_text) > 0:
            x, y0, dy = 5, 25, 30
            cnt = 0
            for key, value in user_text.items():
                text = f'{key}: {value}'
                y = y0 + cnt * dy
                if y > 200 and h < w/1.5 and add_blank:
                    cnt = 0
                    x = w // 2
                    y = y0 + cnt * dy
                cnt += 1
                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, COLORS[text_color], 2)
        return image

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, size=None):
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
