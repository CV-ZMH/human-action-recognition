import os
import json
from PIL import Image
from collections import OrderedDict
import torch
import torch2trt
import numpy as np
import torchvision.transforms as transforms
from trt_pose import models, coco
from trt_pose.parse_objects import ParseObjects


class TrtPose:
    """trtpose estimation"""

    _params = OrderedDict(
            json='human_pose.json',
            weight='weights/pose_estimation/trtpose/densenet121_baseline_att_256x256_B_epoch_160_trt_1.4.0+cu100.pth',
            backbone='densenet121',
            is_trt=True,
            cmap_threshold=0.1,
            link_threshold=0.1
            )

    def __init__(self, size, **kwargs):
        self.__dict__.update(self._params)
        self.__dict__.update(kwargs)

        if not isinstance(size, (tuple, list)):
            size = (size, size)
        if isinstance(self.weight, (tuple, list)):
            self.weight = os.path.join(*self.weight)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.width, self.height = size

        # load humanpose json data
        self.meta = self.load_json(self.json)
        self.topology = coco.coco_category_to_topology(self.meta)
        self.parse_objects = ParseObjects(self.topology, cmap_threshold=self.cmap_threshold, link_threshold=self.link_threshold)

        # load is_trt model
        if self.is_trt:
            self.model  = self._load_trt_model(self.weight)
        else:
            self.model = self._load_torch_model(self.weight, backbone=self.backbone)

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
        """load converted tensorRT model"""

        print(f'[INFO] Loading tensorrt trtpose model : {model_file}')
        model_trt = torch2trt.TRTModule()
        model_trt.load_state_dict(torch.load(model_file))
        model_trt.eval()
        return model_trt

    def _load_torch_model(self, model_file, backbone='densenet121'):
        """load pytorch model with resnet18 encoder or densenet121"""

        print(f'[INFO] Loading pytorch trtpose model with "{model_file}"')
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

        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        return model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        """predict pose estimation on rgb image"""

        img_h, img_w = image.shape[:2]
        self._scale_h = 1.0 * img_h / img_w
        pil_img, tensor_img = self.preprocess(image)
        # print(tensor_img.shape)
        cmap, paf = self.model(tensor_img)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf) # cmap threhold=0.15, link_threshold=0.15
        # all_keypoints = self.get_keypoints(objects, counts, peaks)
        # print('[INFO] Numbers of person detected : {} '.format(counts.shape[0]))
        return counts, objects, peaks

    def preprocess(self, image):
        """Resize image and transform to tensor image"""

        assert isinstance(image, np.ndarray), 'image type need to be array'
        image = Image.fromarray(image).resize((self.width, self.height),
                                              resample=Image.BILINEAR)
        tensor = self.transforms(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        return image, tensor

    @staticmethod
    def get_keypoints(humans, counts, peaks):
        """Get all persons keypoint"""

        all_keypoints = np.zeros((counts, 18, 3), dtype=np.float64) #  counts contain num_persons
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

    def keypoints_to_skeletons_list(self, all_keypoints, scale_h = None):
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
        skeletons_list = []
        NaN = 0
        for keypoints in all_keypoints:
            skeleton = [NaN]*(18*2)
            for idx, kp in enumerate(keypoints):
                skeleton[2*idx] = kp[1]
                skeleton[2*idx+1] = kp[2] * scale_h
            skeletons_list.append(skeleton)
        return skeletons_list, scale_h

    @staticmethod
    def remove_persons_with_few_joints(all_keypoints, min_total_joints=5, min_leg_joints=2, include_head=False):
        """Remove bad skeletons before sending to the tracker"""

        good_keypoints = []
        for keypoints in all_keypoints:
            # include head point or not
            total_keypoints = keypoints[5:, 1:] if not include_head else keypoints[:, 1:]
            num_valid_joints = sum(total_keypoints!=0)[0] # number of valid joints
            num_leg_joints = sum(total_keypoints[-7:-1]!=0)[0] # number of joints for legs

            if num_valid_joints >= min_total_joints and num_leg_joints >= min_leg_joints:
                good_keypoints.append(keypoints)
        return np.array(good_keypoints)


if __name__ == '__main__':
    # import os
    # os.chdir('')
    size = 512
    weight = '../../../../weights/pose_estimation/trtpose/densenet121_baseline_att_512x512_B_epoch_160_trt.pth'
    pose_estimator = TrtPose(size=size, weight=weight)
    x = np.ones((size, size, 3), dtype=np.uint8)
    y = pose_estimator.predict(x)
    # print(y.shape)