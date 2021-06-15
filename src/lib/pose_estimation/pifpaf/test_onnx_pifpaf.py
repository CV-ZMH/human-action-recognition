# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import openpifpaf
import onnx
import onnxruntime as ort
import torch

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))
    openpifpaf.network.Factory().cli(parser)
    return parser.parse_args()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_model(args):
    openpifpaf.network.Factory.configure(args)
    return openpifpaf.network.Factory().factory()[0]


if __name__ == '__main__':
    args = parser()
    args.checkpoint = '/home/zmh/hdd/Custom_Projects/action_recognition/my_action_recogn_dev/weights/pose_estimation/pifpaf/resnet50-210224-202010-cocokp-o10s-d020d7f1.pkl'
    args.W, args.H = (321, 193)

    # paths
    root = '/home/zmh/hdd/Custom_Projects/action_recognition/my_action_recogn_dev'
    img_file = os.path.join(root, 'test_data', '01014.jpg')
    onnx_path = '/home/zmh/hdd/Custom_Projects/action_recognition/my_action_recogn_dev/weights/pose_estimation/pifpaf/openpifpaf-shufflenetv2k30-321x193.onnx'

    # pytorch model
    pifpaf_model = get_model(args)
    # dummy input, BCHW
    dummy_input = torch.randn(1, 3, args.H, args.W)

    # load and check onnx model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # load image
    img_bgr = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # predict with onnxruntime
    ort_sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    ort_inp =  {ort_sess.get_inputs()[0].name : to_numpy(dummy_input)}
    ort_outs = ort_sess.run(None, ort_inp)

