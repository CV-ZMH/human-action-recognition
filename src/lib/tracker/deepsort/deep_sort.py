import cv2
import numpy as np
import torch

from .sort.detection import Detection
from .sort.tracker import Tracker
from .get_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True, **kwargs):

        self.nms_max_overlap = 1.0
        self.extractor = Extractor(**kwargs)
        metric = NearestNeighborDistanceMetric(
            "cosine", max_dist, nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init
        )

    def update(self, bbox_tlwh, ori_img, debug=False):
        """Update tracker state via analyis of current keypoint's bboxes with previous tracked bbox.
        args:
            bbox_tlwh (list): list of keypoints bboxes, (xmin, ymin, w, h)
            img (np.ndarray): original rgb image
        return:
            matches (list[dict]): list of tracking result containing dict as below
                - track_id (int): track id
                - track_bbox (list[float]): tracked bbox (xmin,ymin,xmax,ymax)
                - detection_index (int): keypoint bbox index, useful for drawing keypoints.
        """

        self.height, self.width = ori_img.shape[:2]
        # generate detections
        bbox_tlbr = self.tlwh_to_tlbr(bbox_tlwh)
        features = self._get_features(bbox_tlbr, ori_img)
        detections = [Detection(bbox, features[i]) for i, bbox in enumerate(bbox_tlwh)]

        self.tracker.matches = []
        # update tracker
        self.tracker.predict() # update track_id's time_since_update and age increasement
        self.tracker.update(detections)

        if debug:
            debug_img = self.debug_bboxes(ori_img, self.tracker.tracks, bbox_tlbr)
            return self.tracker.matches, debug_img

        return self.tracker.matches, None

    def increment_ages(self):
        self.tracker.increment_ages()

    def _get_features(self, bbox_tlbr, ori_img):
        im_crops = []
        for box in bbox_tlbr:
            x1, y1, x2, y2 = map(int, box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    @staticmethod
    def tlwh_to_tlbr(bbox_tlwh):
        if isinstance(bbox_tlwh, np.ndarray):
            bbox_tlbr = bbox_tlwh.copy()
        elif isinstance(bbox_tlwh, torch.Tensor):
            bbox_tlbr = bbox_tlwh.clone()

        bbox_tlbr[:, 2] += bbox_tlwh[:, 0]
        bbox_tlbr[:, 3] += bbox_tlwh[:, 1]
        return bbox_tlbr

    @staticmethod
    def debug_bboxes(image, tracks, detections):
        state = {1: 'not-sure', 2: 'sure', 3: 'delete'}
        img = image.copy()

        for track in tracks:
            # if track.is_comfirmed: continue
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            text = f'{track.track_id}: {state[track.state]}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, text, ((x1+x2)//2, (y1+y2)//2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(img, f'det: {idx}', (x1,y1-5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)
        return img