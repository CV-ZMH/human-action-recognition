# -*- coding: utf-8 -*-
import _init_paths
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
    ## IO model files
    weight_file = cfg.TRTPOSE_TORCH.weight
    output_file = weight_file[:-4] + f'_trt_{cfg.TRTPOSE_TORCH.size}.pth'

    trtpose = TrtPose(**cfg.TRTPOSE_TORCH)
    converter = ModelConverter(trtpose)
    converter.convert_trt(trtpose.model, output_file)

if __name__ == '__main__':
    main()
