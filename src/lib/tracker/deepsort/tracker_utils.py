import time

import torch
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from scipy.stats import multivariate_normal
from .get_reid import get_reid_network

def get_data_meta(dataset):
    data_meta = {}
    if dataset.lower() == 'market1501':
        data_meta['num_classes'] = 751
        data_meta['mean'] = [0.1736, 0.1613, 0.1606]
        data_meta['std'] = [0.1523, 0.1429, 0.1443]
    elif dataset.lower() == 'mars':
        data_meta['num_classes'] = 625
        data_meta['mean'] = [0.4144, 0.3932, 0.3824]
        data_meta['std'] = [0.1929, 0.1880, 0.1849]
    else:
        raise NotImplementedError('unknown dataset')
    data_meta['size'] = 256, 128
    return data_meta

def load_reid_model(reid_net, model_path, data_meta):
    print(f'[INFO] Loading reid model: {model_path}')
    model = get_reid_network(reid_net, data_meta['num_classes'], reid=True)
    state_dict = torch.load(model_path, map_location='cpu')['net_dict']
    model.load_state_dict(state_dict)
    model.eval() # add eval() for inference mode
    return model

def get_transform(data_meta):
    tfms = transforms.Compose([
        transforms.Resize(data_meta['size']),
        transforms.ToTensor(),
        transforms.Normalize(data_meta['mean'], data_meta['std'])
    ])
    return tfms

def check_onnx_export(torch_model, input_data, onnx_path):
    '''Compare with pytorch model prediction on input data'''

    start_torch = time.time()
    with torch.no_grad():
        torch_pred = torch_model(input_data)
    end_torch = time.time() - start_torch

    ort_sess = ort.InferenceSession(onnx_path)
    start_onnx = time.time()
    onnx_pred = ort_sess.run(None, {'input': input_data.cpu().numpy()})
    end_onnx = time.time() - start_onnx
    test_near(torch_pred, onnx_pred)
    print("means: {}, {}".format(torch_pred[0].mean().item(), onnx_pred[0].mean()))
    print("Torch speed: {}s,\nONNX speed: {}s".format(end_torch, end_onnx))


# def check_tensorrt_export(torch_model, input_data):
#     '''Check trt model and Compare with pytorch model prediction on input data'''

def test(a, b, cmp, cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-04, atol=1e-06)

def test_near(a, b):
    to_tensor = lambda x: torch.as_tensor(x, device='cpu') \
        if not isinstance(x, torch.Tensor) else x.cpu()
    test(to_tensor(a), to_tensor(b), near)

def get_gaussian_mask(H, W):
	# We will be using 256x128 patch instead of original 128x128 path because we are using for pedestrain with 1:2 AR.
	x, y = np.mgrid[0:1.0:complex(H), 0:1.0:complex(W)] #128 is input size.
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.28,0.28])
	covariance = np.diag(sigma**2)
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
	z = z.reshape(x.shape)
	z = z / z.max()
	z  = z.astype(np.float32)
	mask = torch.from_numpy(z)
	return mask
