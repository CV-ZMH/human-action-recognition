import json
from collections import OrderedDict
import cv2
import torch
import torch2trt
import numpy as np
import torchvision.transforms as transforms
from trt_pose import models, coco
from PIL import Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects


class Pose:
    """
    2D pose estimation
    """
    params = OrderedDict(
            json='pose2d/human_pose.json',
            weight='pose2d/weights/densenet121_baseline_att_256x256_B_epoch_160_trt_1.4.0+cu100.pth',
            backbone='densenet121',
            is_trt=True,
            cmap_threshold=0.1,
            link_threshold=0.1
            )
    JointPairs = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 17), # head
                  (17, 5), (17, 6), (5, 7), (6, 8), (7, 9), (8, 10), # arms
                  (17, 11), (17, 12), (11, 13), (12, 14), (13, 15), (14, 16), # legs
                  (3, 5), (4, 6) # ear to arms
                  ]
    JointPairsRender = JointPairs[:-2]
    JointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    @classmethod
    def _check_kwargs(cls, kwargs):
        for n in kwargs:
            assert n in cls.params.keys(), f'Unrecognized attribute name : "{n}"'

    def __init__(self, size, **kwargs):
        self._check_kwargs(kwargs)
        self.__dict__.update(self.params)
        self.__dict__.update(kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not isinstance(size, tuple):
            size = (size, size)
        self.width, self.height = size
        # load humanpose json data
        self.meta = self.load_json(self.json)
        # load is_trt model
        if self.is_trt:
            self.model  = self._load_trt_model(self.weight)
        else:
            self.model = self._load_torch_model(self.weight, backbone=self.backbone)

        self.topology = coco.coco_category_to_topology(self.meta)
        self.parse_objects = ParseObjects(self.topology, cmap_threshold=self.cmap_threshold, link_threshold=self.link_threshold)
        self.draw_objects = DrawObjects(self.topology)

        # transformer
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    @staticmethod
    def load_json(json_file):
        with open(json_file, 'r') as f:
            meta = json.load(f)
        return meta

    def _load_trt_model(self, model_file):
        """
        load converted tensorRT model
        """
        print('[INFO] Loading tensorrt pose model')
        model_trt = torch2trt.TRTModule()
        model_trt.load_state_dict(torch.load(model_file))
        model_trt.eval()
        return model_trt

    def _load_torch_model(self, model_file, backbone='densenet121'):
        """
        load pytorch model with resnet18 encoder or densenet121
        """
        print('[INFO] Loading pytorch 2d_pose model with "{}"'.format(backbone.title()))
        num_parts = len(self.meta['keypoints'])
        num_links = len(self.meta['skeleton'])

        if backbone=='resnet18':
            model = models.resnet18_baseline_att(cmap_channels=num_parts,
                                                 paf_channels=2 * num_links)
        elif backbone=='densenet121':
            model = models.densenet121_baseline_att(cmap_channels=num_parts,
                                                    paf_channels= 2 * num_links)
        else:
            print('not supported model type "{}"'.format(backbone))
            return -1

        model.to(self.device).eval()
        model.load_state_dict(torch.load(model_file))
        return model

    def predict(self, image: np.ndarray):
        """
        predict pose estimation on rgb array image
        *Note - image need to be RGB numpy array format
        """
        self.img_h, self.img_w = image.shape[:2]
        self._scale_h = 1.0 * self.img_h / self.img_w
        pil_img, tensor_img = self.preprocess(image)
        # print(tensor_img.shape)
        cmap, paf = self.model(tensor_img)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf) # cmap threhold=0.15, link_threshold=0.15
        all_keypoints = self.get_keypoints(objects, counts, peaks)
        # print('[INFO] Numbers of person detected : {} '.format(counts.shape[0]))
        return all_keypoints

    def preprocess(self, image):
        """
        resize image and transform to tensor image
        """
        assert isinstance(image, np.ndarray), 'image type need to be array'
        image = Image.fromarray(image).resize((self.width, self.height),
                                              resample=Image.BILINEAR)
        tensor = self.transforms(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        return image, tensor

    @staticmethod
    def get_keypoints(humans, counts, peaks):
        """
        Get all persons keypoint
        """
        all_keypoints = np.zeros((counts, 18, 3), dtype=np.float32) #  counts contain num_persons
        for count in range(counts):

            human = humans[0][count]
            C = human.shape[0]
            for j in range(C):
                k = int(human[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    peak = (j, float(peak[1]), float(peak[0]))
                    all_keypoints[count, j] = peak
                else:
                    peak = (j, 0., 0.)
                    all_keypoints[count, j] = peak

        return all_keypoints

    def keypoints_to_skels_list(self, all_keypoints, scale_h = None):
        ''' Get skeleton data of (x, y * scale_h) from humans.
        Arguments:
            humans {a class returned by self.detect}
            scale_h {float}: scale each skeleton's y coordinate (height) value.
                Default: (image_height / image_widht).
        Returns:
            skeletons {list of list}: a list of skeleton.
                Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
            scale_h {float}: The resultant height(y coordinate) range.
                The x coordinate is between [0, 1].
                The y coordinate is between [0, scale_h]
        '''
        if scale_h is None:
            scale_h = self._scale_h
        skeletons = []
        NaN = 0
        for keypoints in all_keypoints:
            skeleton = [NaN]*(18*2)
            for idx, kp in enumerate(keypoints):
                skeleton[2*idx] = kp[1]
                skeleton[2*idx+1] = kp[2] * scale_h
            skeletons.append(skeleton)
        return skeletons, scale_h

    def draw2D(self, image, all_keypoints, draw_circle=False, draw_numbers=False, skip_from=-1):
        """
        draw all persons keypoints
        """
        thickness = 3 if image.shape[1]/640 > 1.3 else 1
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
                    cv2.circle(image, center, thickness, (0, 0, 0), thickness+2)
                if draw_numbers:
                    cv2.putText(image, str(int(kp[0])), center, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            0.6 if thickness==1 else 1, (0,0,255), thickness)

             # draw line on image
            for pair_order, pair in enumerate(self.JointPairsRender):
                if pair[0] < skip_from or pair[1] < skip_from: continue
                if pair[0] in visibilities or pair[1] in visibilities: continue
                cv2.line(image, centers[pair[0]], centers[pair[1]], self.JointColors[pair_order], 3)

    @staticmethod
    def remove_persons_with_few_joints(all_keypoints):
        """ Remove bad skeletons before sending to the tracker"""
        good_keypoints = []
        for keypoints in all_keypoints:
            no_head_keypoints = keypoints[5:, 1:]
            num_valid_joints = sum(no_head_keypoints!=0)[0] # number of valid joints (withoud head)
            num_leg_joints = sum(no_head_keypoints[-7:-1]!=0)[0] # number of joints for legs

            if num_valid_joints >= 5 and num_leg_joints >= 0:
                good_keypoints.append(keypoints)
        return np.array(good_keypoints)

if __name__ == '__main__':
    import os
    os.chdir('../')
    size = 512
    pose = Pose(size=size, weight='pose2d/weights/densenet121_baseline_att_512x512_B_epoch_160_trt.pth')
    x = np.ones((size, size, 3), dtype=np.uint8)
    y = pose.predict(x)
    print(y.shape)
