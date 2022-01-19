import sys
sys.path.append('../src/lib')
import os
import torch
import torch2trt
from fire import Fire
from utils.config import Config
from pose_estimation.trtpose.trtpose import TrtPose

class ExportTrt(TrtPose):
    """Convert trtpose pytorch model to tensorrt."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, save_name: str=None):
        inp_shape = (1, 3, self.height, self.width)
        print(f'[INFO] Converting to tensorrt with input shape: {inp_shape}')
        data = torch.zeros(inp_shape).to(self.device)

        model_trt = torch2trt.torch2trt(
            self.model,
            [data],
            fp16_mode=True,
            max_workspace_size=1<<25
        )
        folder, filename = os.path.split(self.model_path)
        if save_name is None:
            save_name = f'{filename[:-4]}_{self.height}x{self.width}.trt'
        output_path = os.path.join(folder, save_name)
        torch.save(model_trt.state_dict(), output_path)
        print('[INFO] Saved to {}'.format(output_path))

def main(config, save_name=None):
    """Convert pytorch pose model to tensorrt.
    config : must be trtpose config file. you can use 'infer_trtpose_deepsort_dnn.yaml' this also.
    save_name : save tensorrt name, default(None)
    """

    cfg = Config(config)
    pose_kwargs = cfg.POSE
    model_path = pose_kwargs.model_path
    model_path = model_path if not isinstance(model_path, (tuple, list)) else os.path.join(*model_path)

    assert not model_path.endswith(".trt"), f'model file must be pytorch weight, \n{model_path}'
    export_trt = ExportTrt(**pose_kwargs)
    export_trt(save_name)

if __name__ == '__main__':
    Fire(main)
    # main('../configs/infer_trtpose_deepsort_dnn.yaml')
