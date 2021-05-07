# Simple Multi-Person Human Activity Recognition  

This is the multi person action recognition pipeline with pose estimation as `trt_pose`, tracking as `deepsort` and action classifier as `simple dnn classifier` .

Original [repo](https://github.com/felixchenfy/Realtime-Action-Recognition) is with `tfpose`, tracking based on `euclidean distance of skeletons motions` and action classifier as `simple dnn classifier`.  


> 9 actions : ['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']

|<img src="assets/aung_la.gif"  width="480" height="320"/> |
|:--:|
| __Multi Person Activity Analysis Demo__ |

## Table of Contents

- [1. Installation](#1-installation)
- [2. Quick Demo](#2-quick-demo)
- [3. Training Action Recognition](#3-trainingg-action-recognition)
- [4. References](#4-references)
---
## 1. Installation

Firtst, Python >= 3.6

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

Run this command to install them.
```bash
pip install -r requirements.txt
```
---

## 2. Quick Demo

- Download the pretrained weight files to run the demo.

| Model | Weight |
|---|---|
| *Trt_Pose* | [densenet121_backbone](https://drive.google.com/open?id=13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU) |
| *Deepsort*| [reid.pth](https://drive.google.com/drive/u/0/folders/1EQ847gmKlktb9lOLMQRQvA7g708Nkdkt)|

- Then put them to these folder
    - *trt_pose* weight to `weights/pose_estimation/trtpose`.
    - *deepsort* weight to `weights/tracker/deepsort/`

- Then convert the *trt_pose* weight for `tensorrt` model.
```bash
# check the IO weight file in configs/trtpose.yaml
cd src && python convert_model.py
```

- Before running `demo.py`,  you need to check the config files [confiigs/inference_config.yaml](confiigs/inference_config.yaml) and [configs/trtpose.yaml](configs/trtpose.yaml)
- Run **action recogniiton**.
```bash
cd src
python demo.py --mode action --src ../test_data/fun_theory.mp4 --save_path ../output
```

- Run **person skeleton tracking**.
```bash
cd src
python demo.py --mode track --src ../test_data/fun_theory.mp4 --save_path ../output
```
---

## 3. Training Action Recognition

- Download the sample training dataset from [original repo](https://drive.google.com/open?id=1V8rQ5QR5q5zn1NHJhhf-6xIeDdXVtYs9)

- Modify the [data_root](https://github.com/CV-ZMH/human_activity_recognition/blob/d5c1d25b62c2147994d06ed3eda12a85b03ceeef/configs/training_config.yaml#L5) and [extract_path](https://github.com/CV-ZMH/human_activity_recognition/blob/d5c1d25b62c2147994d06ed3eda12a85b03ceeef/configs/training_config.yaml#L6) with your IO dataset path in [configs/training_config.yaml](configs/training_config.yaml).

- Depend on your need, you may change [configs/training_config.yaml](configs/training_config.yaml).

- Then run training
```bash
./run_training_trtpose_action.sh
```
---
## 4. References
Most of the action recognition training and features extraction code are inspired from  the [realtime action recognition](https://github.com/felixchenfy/Realtime-Action-Recognition) repo.

Pose Estimation codes are inspired from [nvidia trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/README.md) repo.

 Tracking codes are mainly references from [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) repo.

## To Do

- [ ] add more trackers like fair-mot
- [ ] add tr_tpose estimation training code
- [ ] add more action classifier based on LSTM
- [ ] add speed measurement for realtime
