# -*- coding: utf-8 -*-
from pathlib import Path
import torch2trt
import torch
import argparse

def options():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', default='weights/densenet121_baseline_att_256x256_B_epoch_160.pth',
                    help='model file path')
    ap.add_argument('--backbone', default='densenet121', help='backbone name')
    ap.add_argument('--output', default='weights', help='output model file')
    return ap.parse_args()


class ModelConverter():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def convert_trt(self, model, output: str, w=224, h=224):
        """
        convert pytorch model to tensorrt
        """
        data = torch.zeros((1, 3, h, w)).to(self.device)
        print('[INFO] Converting to tensorrt')
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt.state_dict(), output)
        print('[INFO] Saved to {}'.format(output))

def main():
    from pose2d import Pose2D
    converter = ModelConverter()
    args = options()
    pose = Pose2D(weight=args.weight, json='weights/human_pose.json', is_trt=False)
    converter = ModelConverter()
    weight = Path(args.weight)
    output = Path(args.output)/f'{weight.stem}_trt_{torch.__version__}.pth'
    converter.convert_trt(pose.model, output, w=pose.width, h=pose.height)

if __name__ == '__main__':
    main()
