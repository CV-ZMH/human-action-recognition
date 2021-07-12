import os
from fire import Fire
import tensorrt as trt

def onnx2trt(onnx_path, mode='fp16', max_batch=100, calib=None, save_path=None):
    """Convert onnx model to tensorrt engine, use mode of ['fp32', 'fp16', 'int8'].
    return : trt engine
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # tensorrt 7 need to set explicit batch
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

        # allow TensorRT to use up to 1GB of GPU memory for tactic selection
        builder.max_workspace_size = 3*1 << 30
        builder.max_batch_size = max_batch

        if mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), 'not support int8'
            builder.int8_mode = True
            builder.int8_calibrator = calib

        if mode.lower() == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), 'not support fp16'
            builder.fp16_mode = True

        # generate TensorRT engine optimized for the target platform
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 256, 128), # min
                          (max_batch//2, 3, 256, 128), # optimize
                          (max_batch, 3, 256, 128)) # max
        config.add_optimization_profile(profile)

        # parse ONNX
        print(f'[INFO] Parsing onnx: {onnx_path}')
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for e in range(parser.num_errors):
                    print(parser.get_error(e))
                raise TypeError("Parser parse failed.")


        print('[INFO] Building trt engine...')
        engine = builder.build_engine(network, config)
        if save_path is None:
            save_path = onnx_path.replace('onnx', 'trt')
        print(f'[INFO] Saving engine file: {save_path}')
        with open(save_path, 'wb') as f:
            f.write(engine.serialize())

def test():
    onnx_path = '/home/zmh/Desktop/HDD/Workspace/dev/human-action-recognition/weights/tracker/deepsort/siamese_mars.onnx'
    mode = 'fp16'
    max_batch = 100
    calib = None
    save_path = None
    onnx2trt(onnx_path, mode, max_batch, calib, save_path)

if __name__ == '__main__':
    Fire(onnx2trt)
    # test()
