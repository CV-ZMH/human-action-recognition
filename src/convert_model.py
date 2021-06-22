# -*- coding: utf-8 -*-
import _init_paths
import os
import torch
import torch2trt
from fire import Fire
from utils.config import Config
from lib.pose_estimation.trtpose.trtpose import TrtPose

class ExportTrt(TrtPose):
    """Convert trtpose pytorch model to tensorrt"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, save_name: str=None):
        inp_shape = (1, 3, self.height, self.width)
        print('[INFO] Converting to tensorrt with input shape: {inp_shape}')
        data = torch.zeros(inp_shape).to(self.device)
        model_trt = torch2trt.torch2trt(self.model,
                                        [data],
                                        fp16_mode=False,
                                        max_workspace_size=1<<25)
        folder, filename = os.path.split(self.weight)
        if save_name is None:
            save_name = f'{filename[:-4]}_{self.height}x{self.width}.trt'
        output_path = os.path.join(folder, save_name)
        torch.save(model_trt.state_dict(), output_path)
        print('[INFO] Saved to {}'.format(output_path))

def main(cfg_file, save_name=None):
    """Convert pytorch pose model to tensorrt
    cfg_file : trtpose config file
    save_name : save tensorrt name, default(None)
    """

    cfg = Config(config_file=cfg_file)
    trtpose_cfg = cfg.TRTPOSE
    export_trt = ExportTrt(**trtpose_cfg, **trtpose_cfg.torch_model)
    export_trt(save_name)

if __name__ == '__main__':
    Fire(main)