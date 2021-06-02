# -*- coding: utf-8 -*-
import os
import time
import pickle
from collections import OrderedDict
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import openpifpaf
import tensorrt as trt
import pycuda.driver as cuda
import openpifpaf.decoder.cifcaf as OriginalDecoder


class CifCafDecoder():
    def __init__(self):
        self.cif_metas = pickle.load(open( "cif_metas.pkl", "rb" ))
        self.caf_metas = pickle.load(open( "caf_metas.pkl", "rb" ))
        self._decoder = OriginalDecoder.CifCaf(cif_metas = self.cif_metas,
                                                caf_metas = self.caf_metas)

    def decode(self, fields):
        return self._decoder(fields)


def allocate_buffers(engine):
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    for i in range(engine.num_bindings):
        binding = engine[i]
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    stream = cuda.Stream()  # create a CUDA stream to run inference
    return bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream


class PifPafPose:
    """trtpose estimation"""

    _params = OrderedDict(
            json='human_pose.json',
            weight='',
            backbone='densenet121',
            is_trt=True,
            cmap_threshold=0.1,
            link_threshold=0.1
            )

    @classmethod
    def _check_kwargs(cls, kwargs):
        for n in kwargs:
            assert n in cls._params.keys(), f'Unrecognized attribute name : "{n}"'

    def __init__(self, size, **kwargs):
        self._check_kwargs(kwargs)
        self.__dict__.update(self._params)
        self.__dict__.update(kwargs)

        if not isinstance(size, (tuple, list)):
            size = (size, size)
        self.width, self.height = size
        self.cuda_context = None

        self._init_cuda_stuff()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def _load_engine(self, weight):
        trt_logger = trt.Logger(trt.Logger.INFO)
        with open(weight, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _init_cuda_stuff(self):
        cuda.init()
        self.engine = self._load_engine()
        self.device = cuda.Device(0)  # enter your Gpu id here
        self.cuda_context = self.device.make_context()
        self.engine_context = self.engine.create_execution_context()
        bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream = allocate_buffers(self.engine)
        self.bindings = bindings
        self.host_inputs = host_inputs
        self.host_outputs = host_outputs
        self.cuda_inputs = cuda_inputs
        self.cuda_outputs = cuda_outputs
        self.stream = stream

    def __del__(self):
        """Free cuda memory"""
        self.cuda_context.pop()
        del self.cuda_context
        del self.engine_context
        del self.engine

    def predict(self, image: np.ndarray):
        """predict pose estimation on rgb image"""
        image = cv2.resize(image, (self.width, self.height))
        pil_im = Image.fromarray(image)
        preprocess = None

        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)
        loader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=False, pin_memory=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

        for images_batch, _, __ in loader:
            np_img = images_batch.numpy()


        bindings = self.bindings
        host_inputs = self.host_inputs
        host_outputs = self.host_outputs
        cuda_inputs = self.cuda_inputs
        cuda_outputs = self.cuda_outputs
        stream = self.stream

        host_inputs[0] = np.ravel(np.zeros_like(np_img))

        self.cuda_context.push()
        t_begin = time.perf_counter()

        np.copyto(host_inputs[0], np.ravel(np_img))
        cuda.memcpy_htod_async(
            cuda_inputs[0], host_inputs[0], stream)

        self.engine_context.execute_async(
            batch_size=1,
            bindings=bindings,
            stream_handle=stream.handle)
        cif=[None] * 1
        caf=[None] * 1
        cif_names=['cif']
        caf_names=['caf']
        for i in range(1, self.engine.num_bindings):
            cuda.memcpy_dtoh_async(
                host_outputs[i - 1], cuda_outputs[i - 1], stream)

        stream.synchronize()

        for i in range(1, self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            name = self.engine.get_binding_name(i)
            total_shape = np.prod(shape)
            output = host_outputs[i - 1][0: total_shape]
            output = np.reshape(output, tuple(shape))
            if name in cif_names:
                index_n = cif_names.index(name)
                tmp = torch.from_numpy(output[0])
                cif = tmp.cpu().numpy()
            elif name in caf_names:
                index_n = caf_names.index(name)
                tmp = torch.from_numpy(output[0])
                caf = tmp.cpu().numpy()
        heads = [cif, caf]
        self.cuda_context.pop()
        inference_time = time.perf_counter() - t_begin
        fields = heads
        return fields

    def preprocess(self, image):
        """Resize image and transform to tensor image"""
        assert isinstance(image, np.ndarray), 'image type need to be array'
        image = Image.fromarray(image).resize((self.width, self.height),
                                              resample=Image.BILINEAR)
        tensor = self.transforms(image)
        tensor = tensor.unsqueeze(0)
        return image, tensor

if __name__ == '__main__':
    size = (321, 193)
    root = '/home/zmh/hdd/Custom_Projects/action_recognition/my_action_recogn_dev'
    img_file = os.path.join(root, 'test_data', '01014.jpg')
    weight= os.path.join(root, 'weights/pose_estimation/pifpaf/openpifpaf-shufflenetv2k30-321x193.trt')

    img_bgr = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    decoder = CifCafDecoder()
    pose_estimator = PifPafPose(size=size, weight=weight)
    heads = pose_estimator.predict(img_rgb)