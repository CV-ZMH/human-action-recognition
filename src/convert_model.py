# -*- coding: utf-8 -*-
import _init_paths
import os
import torch
import torch2trt
from utils import parser
from pose_estimation import TrtPose

class ModelConverter():
    """Convert trtpose pytorch model to tensorrt"""

    def __init__(self, trtpose):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = trtpose.model
        self.width, self.height = trtpose.width, trtpose.height

    def convert_trt(self, model, output: str):

        print('[INFO] Converting to tensorrt')
        data = torch.zeros((1, 3, self.height, self.width)).to(self.device)
        model_trt = torch2trt.torch2trt(model,
                                        [data],
                                        fp16_mode=True,
                                        max_workspace_size=1<<25)

        torch.save(model_trt.state_dict(), output)
        print('[INFO] Saved to {}'.format(output))

def main():
    # settings
    cfg = parser.YamlParser(config_file='../configs/trtpose.yaml')
    output_name = "_".join(map(str, cfg.TRT_CFG.weight))
    output_path = os.path.join(cfg.weight_folder, output_name)
    cfg.TORCH_CFG.weight = os.path.join(cfg.weight_folder, cfg.TORCH_CFG.weight)

    trtpose = TrtPose(**cfg.TRTPOSE, **cfg.TORCH_CFG)
    converter = ModelConverter(trtpose)
    converter.convert_trt(trtpose.model, output_path)

if __name__ == '__main__':
    main()
