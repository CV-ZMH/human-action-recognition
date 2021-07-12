import sys
sys.path.append('../src/lib')
import os.path as osp

import onnx
import torch
from fire import Fire
from PIL import Image
from tracker.deepsort import tracker_utils as utils

IMG_PATH = '0003C5T0011F003.jpg'

def make_output_path(model_path):
    folder, filename = osp.split(model_path)
    save_filename = f'{filename[:-4]}.onnx'
    return osp.join(folder, save_filename)

def get_input_data(data_meta, img_path, bs=10):
    '''Load the input data and preprocess it'''
    image = Image.open(img_path)
    tfms = utils.get_transform(data_meta)
    input_data = tfms(image)[None]
    return input_data

def export_onnx(
        model_path,
        reid_name,
        dataset_name,
        output_path=None,
        check=False
    ):

    # get input data
    data_meta = utils.get_data_meta(dataset_name)
    input_data = get_input_data(data_meta, IMG_PATH)
    # load pytorch model
    model = utils.load_reid_model(reid_name, model_path, data_meta).to('cpu')
    # export to onnx
    if output_path is None:
        output_path = make_output_path(model_path)
    print(f'[INFO] Converting to onnx: {output_path}')
    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}} # dynamic batch size
    torch.onnx.export(
        model,
        input_data,
        output_path,
        # verbose=False,  # store the trained parameter weights inside the model file
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        export_params=True
    )

    print('[INFO] Testing with onnx checker')
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print('Ok')

    if check:
        utils.check_onnx_export(model, input_data, output_path)

def test():
    model_path = '/home/zmh/Desktop/HDD/Workspace/dev/human-action-recognition/weights/tracker/deepsort/siamese_mars.pth'
    reid_name = 'siamesenet'
    dataset_name = 'mars'
    export_onnx(model_path, reid_name, dataset_name, check=True)


if __name__ == '__main__':
    Fire(export_onnx)
    #test()
