# Simple Real Time Multi Person Action Recognition  

## News
:boom:  Added trained weights of [**siamesenet**](https://github.com/abhyantrika/nanonets_object_tracking/blob/master/siamese_net.py) and [**aligned reID**](https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch) networks and training script. They are used in cosine metric learnings of deep sort pipeline.

:boom: Added debug-tracker flag to `demo.py` script for visualizing tracker bboxes and keypoints bboxes.

:boom: IoU matching step required for tracked bboxes ID and keypoints' bboxes index is directly replaced  with deepsort matching cascade.

:boom: Fixed and cleaned for deepsort bbox input format as *xmin, ymin, xmax, ymax*.

:boom: Added current frame detailed and config parameters in left side of the display.


## Table of Contents
- [1. Overview](#1-overview)
- [2. Inference Speed Comparison](#2-inference-speed)
- [3. Installation](#3-installation)
- [4. Quick Demo](#4-quick-demo)
- [5. Training](#5-training)
    - [5.1 Training for action classifier](#5-1-training-for-action-classifier)
    - [5.2 Training reID network for deepsort](#5-2-training-reid-network-for-deepsort)
- [6. References](#6-references)
---

## Overview
> Pretrained actions, total 9 classes : **['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']**

<!-- |<img src="assets/aung_la.gif"  width="480" height="320"/> |
|:--:|
| Multi Person Action Recognition Demo | -->

This is the 3 steps multi-person action recognition pipeline using
1. pose estimation with [trtpose](https://github.com/NVIDIA-AI-IOT/trt_pose)
2. people tracking with [deepsort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
3. action classifier with [dnn](https://github.com/felixchenfy/Realtime-Action-Recognition#diagram)

You can easily add different pose estimation, tracker and action recognizer by referencing the code structure of the pipeline. I will also add others for better action recognition and tracking result.  

Action classifier is used from [this repo](https://github.com/felixchenfy/Realtime-Action-Recognition#diagram) and his dataset also.

## Inference Speed
Tested PC specification

- **OS**: Ubuntu 18.04
- **CPU**: i7-8750H @ 4.100GHz
- **GPU**:  RTX 2060
- **CUDA**: 10.2
- **TensorRT**: 7.1.3.4

| Pipeline Step | Step's Input Size (H, W) | FPS|
| -  | - | - |
| trtpose  | (256x256) | 25  |
| trtpose + deepsort | (256x256) + (256x128) | 18 |
| trtpose + deepsort + dnn | (256x256) + (256x128) + (-) | 15 |


## 3. Installation

First, Python >= 3.6

### Step 1 - Install Dependencies

Check this [installation guide](https://github.com/CV-ZMH/Installation-Notes-for-Deeplearning-Developement#install-tensorrt) for deep learning packages installation.

Here is required packages for this project and you need to install each of these.
1. Nvidia-driver 450
2. [Cuda-10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal) and [Cudnn 8.0.2](https://developer.nvidia.com/rdp/cudnn-archive)
3. [Pytorch 1.7.1](https://pytorch.org/get-started/previous-versions/) and [Torchvision 0.8.2](https://pytorch.org/get-started/previous-versions/)
4. [TensorRT 7.1.3](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html)

### Step 2 - Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install --plugins
```
### Step 3 - Install trt_pose

```bash
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python setup.py install
```
Other python packages are in [`requirements.txt`](requirements.txt).

Run below command to install them.
```bash
pip install -r requirements.txt
```
---

## 4. Quick Demo

### 4.1- Download the Pretrained Models
Action Classifier Pretrained models are already uploaded in the path `weights/classifier/dnn`.
- Download the pretrained weight files to run the demo.
| Model | Weight |
|---|---|
| *Trt_Pose (densenet121)* | [weight](https://drive.google.com/file/d/1De2VNUArwYbP_aP6wYgDJfPInG0sRYca/view?usp=sharing) |
| *Deepsort (original reid)*| [weight](https://drive.google.com/file/d/1xw7Sv4KhrzXMQVVQ6Pc9QeH4QDpxBfv_/view?usp=sharing)|
| *Deepsort (siamese reid)*| [weight](https://drive.google.com/file/d/11OmfZqnnG4UBOzr05LEKvKmTDF5MOf2H/view?usp=sharing)|
| *Deepsort (align reid)*| [weight](https://drive.google.com/file/d/1WKv1QJFuWNVh_GLzQ_Ssu9XXSNR2s2b5/view?usp=sharing)|

- Then put them to these folder
    - *deepsort* weight to `weights/tracker/deepsort/`
    - *trt_pose* weight to `weights/pose_estimation/trtpose`.

### 4.2- Convert TrTPose to TensorRT (Optional)

If you don't have installed tensorrt on your system, just skip this step. You just need to set trtpose pytorch model path of the [model path]() in the config.

Run the below command to convert  TrtPose pytorch weight to tensorrt.
```bash
# check the IO weight file in configs/trtpose.yaml
cd src
python convert_tensorrt.py --config_file ../configs/infer_trtpose_deepsort_dnn.yaml
```
 *Note*:  original **densenet121_trtpose** model is trained with **256** input size. So, if you want to convert tensorrt model with bigger input size (like 512), you need to change [size]() parameter in `configs/infer_trtpose_deepsort_dnn.yaml` file.

### 4.3- Run Demo.py

Arguments list of `Demo.py`
- **task** [pose, track, action] : Inference mode for testing _Pose Estimation_, _Tracking_ or _Action Recognition_.
- **config** : inference config file path. (default=../configs/inference_config.yaml)
- **source** : video file path to predict action or track. If not provided, it will use *webcam* source as default.
- **save_folder** : save the result video folder path. Output filename format is composed of "_{source video name/webcam}{pose network name}{deepsort}{reid network name}{action classifier name}.avi_". If  not provided, it will not save the result video.
- **draw_kp_numbers** : flag to draw keypoint numbers of each person for visualization.
- **debug_track** : flag to debug tracking for tracker's bbox state and current detected bbox of tracker's inner process with bboxes visualization.

Before running the demo.py, you need to change some parameters in
 [confiigs/infer_trtpose_deepsort_dnn.yaml]() file.

Examples:
-  To use different reid network, [`reid_net`]() and it's [`model_path`]() in `TRACKER` node.
- Set also [`model_path`] of trtpose  in `POSE` node.
- You can also tune other parameters of `TRACKER` node and `POSE` node for better tracking and action recognition result.

Then, Run **action recogniiton**.
```bash
cd src
# for video, use --source flag to your video path
python demo.py --task action --source ../test_data/fun_theory.mp4 --save_folder ../output --debug_track
# for webcam, no need to provid --source flag
python demo.py --task action --save_path ../output --debug_track
```

Run **pose tracking**.
```bash
# for video, use --src flag to your video path
python demo.py --task track --source ../test_data/fun_theory.mp4 --save_path ../output
# for webcam, no need to provid --source flag
python demo.py --task track --save_path ../output
```

Run **pose estimation** only.
```bash
# for video, use --src flag to your video path
python demo.py --task pose --source ../test_data/fun_theory.mp4 --save_path ../output
# for webcam, no need to provid --source flag
python demo.py --task pose --save_path ../output
```
---

## 5. Training

### 5.1 Train for Action Classifier
- For this step, it's almost same preparations as original repo to train action classifier. You can directly reference of dataset preparation, feature extraction information and training information from the [original repo](https://github.com/felixchenfy/Realtime-Action-Recognition).
- Then, Download the sample training dataset from the [original repo](https://drive.google.com/open?id=1V8rQ5QR5q5zn1NHJhhf-6xIeDdXVtYs9)

- Modify the [`data_root`]() and [`extract_path`]() with your IO dataset path in [configs/train_action_recogn_pipeline.yaml](configs/train_action_recogn_pipeline.yaml).

- Depend on your custom action training, you have to change the parameters in  [configs/train_action_recogn_pipeline.yaml](configs/train_action_recogn_pipeline.yaml).

- Then run below command to train action classifier step by step.
```bash
cd src && bash ./train_trtpose_dnn_action.sh
```

### 5.2  Train for Tracker reID Network [optional]
To train different reid network for cosine metric learning used in deepsort:
- Download the reid dataset [Market1501](https://www.kaggle.com/pengcw1/market-1501/data)
- Prepare dataset with this command.
```bash
cd src && python prepare_market1501.py --root (your dataset root)
```
- Modify the [`tune_params`]() for multiple runs to find hyper parameter search as your need.

- Then run below to train reid network.
```bash
cd src && python train_reid.py --config ../configs/train_reid.yaml
```
---
## 6. References
- [realtime action recognition](https://github.com/felixchenfy/Realtime-Action-Recognition)
- [nvidia trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/README.md)
- [DeepSort](https://github.com/nwojke/deep_sort)
- [Pytorch Deepsort YOLOv5](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [Aligned Reid Network](https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch)

## 7. TODO  
- [x] Add FPS of current frame.
- [x] Fix ID matching after matching cascade in DeepSort
- [x] Add different reid network used in DeepSort
- [ ] Add open_pifpaf pose estimation
- [ ] Add norfair tracker
- [ ] Add other action recognition network.
- [ ] Train on different datasets, loss strategies, reid networks for deepsort tracking.
