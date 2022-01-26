import os
import time

import torch
torch.tensor(1, device='cuda') # temp fix for pycuda gpu allocation conflict
import numpy as np
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
except: print("pycuda or tensorrt not installed.")
from PIL import Image
from torchvision import transforms

from . import tracker_utils as utils

def np_transform(image, data_meta):
    h, w = data_meta['size']
    mean, std = data_meta['mean'], data_meta['std']
    image = Image.fromarray(image).resize((w, h), resample=Image.BILINEAR)
    image = np.array(image).transpose((2, 0, 1))
    image = (image / 255).astype('float32')
    image = (image - np.array(mean).reshape(-1,1,1))/ np.array(std).reshape(-1,1,1)
    return image

class FeatureExtractor:
    def __init__(self, model_path, reid_name, dataset_name, verbose=True, **kwargs):
        self.verbose = verbose
        self.reid_name = reid_name
        # get dataset info
        self.data_meta = utils.get_data_meta(dataset_name)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load reid model
        if isinstance(model_path, (tuple, list)):
            model_path = os.path.join(*model_path)
        assert os.path.isfile(model_path), f"reid model don't exist : {model_path}"
        self.is_trt = model_path.endswith('.trt')
        if self.is_trt:
            # self.device = torch.device('cpu')
            self.extractor = self._load_trt_model(model_path)
            self.context = self.extractor.create_execution_context()

        else:
            self.extractor = self._load_torch_model(model_path)
        # setup transforms
        self.tfms = transforms.Compose([
            transforms.Resize(self.data_meta['size']),
            transforms.ToTensor(),
            transforms.Normalize(self.data_meta['mean'], self.data_meta['std'])
        ])

    def _load_trt_model(self, model_path):
        """Load TensorRT engine."""
        if self.verbose: print(f'[INFO] Loading TensorRT reid model: {model_path}.')
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as trt_sess:
            engine = trt_sess.deserialize_cuda_engine(f.read())
        return engine

    def _load_torch_model(self, model_path):
        """Load pytorch reid model."""
        if self.verbose: print(f'[INFO] Loading pytorch reid model: {model_path}.')
        model = utils.get_reid_network(self.reid_name, self.data_meta['num_classes'], reid=True)
        state_dict = torch.load(model_path, map_location='cpu')['net_dict']
        model.load_state_dict(state_dict)
        model.to('cuda').eval() # add eval() for ineference mode
        return model

    def _preprocess(self, images):
        imgs = [self.tfms(Image.fromarray(image))[None] for image in images]
        img_batch = torch.cat(imgs, dim=0).float()
        return img_batch

    def inference_trt(self, batch_images: np.ndarray):
        """Predict with TensorRT engine."""
        # first prepare I/O shape for device allocation
        batch = batch_images.shape[0] # get batch dim of current input
        output_shape = [batch, self.extractor.get_binding_shape(1)[1]]
        output = np.empty(output_shape, dtype=np.float32)
        # allocate device buffer for inputs and outputs
        # import pycuda.autoinit
        d_input = cuda.mem_alloc(batch_images.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        self.context.set_binding_shape(0, batch_images.shape) # set binding shape for dynamic input
        # Transfer input data to gpu
        cuda.memcpy_htod_async(d_input, batch_images, stream)
        # Run Inference
        self.context.execute_async(
            bindings=[int(d_input), int(d_output)],
            stream_handle= stream.handle
        )
        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        return output

    @torch.no_grad()
    def __call__(self, images):
        batch_images = self._preprocess(images)
        if self.is_trt:
            # run with trt engine
            features = self.inference_trt(batch_images.numpy())
        else:
            batch_images = batch_images.to('cuda')
            features = self.extractor(batch_images).cpu().numpy()
        return features

def test():
    trt_model_path = '/home/zmh/Desktop/HDD/Workspace/dev/human-action-recognition/weights/tracker/deepsort/wide_residual_reid_market1501.trt'
    torch_model_path = '/home/zmh/Desktop/HDD/Workspace/dev/human-action-recognition/weights/tracker/deepsort/wide_residual_reid_market1501.pth'

    img_path = '/home/zmh/Desktop/HDD/Workspace/dev/human-action-recognition/weights/tracker/deepsort/0003C5T0011F003.jpg'
    reid_name = 'wideresnet'
    dataset_name = 'market1501'

    import cv2
    input_data = cv2.imread(img_path)[...,::-1][None]

    np.random.seed(123)
    # trt_extractor = FeatureExtractor(trt_model_path, reid_name, dataset_name)
    torch_extractor = FeatureExtractor(torch_model_path, reid_name, dataset_name)
    start = time.time()
    for i in range(150):
        # input_data = [(np.random.rand(256, 128, 3)*255).astype('uint8') for _ in range(i+1)]
        # print(len(input_data))
        # trt_features = trt_extractor(input_data)
        # print(trt_features.mean())

        torch_features = torch_extractor(input_data)
        print(torch_features.mean().item())
    end = time.time() - start
    print('Speed: {} s'.format(end))



if __name__ == "__main__":
    # Fire(fire)
    input_data = test()
