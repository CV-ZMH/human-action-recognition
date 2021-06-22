# -*- coding: utf-8 -*-
from .wide_resnet import WideResnet
from .siamese_net import SiameseNet
from .align_reid import AlignedReid

_reid_model = {
        'wideresnet' : WideResnet,
        'siamesenet' : SiameseNet,
        'align_reid' :  AlignedReid
        }

def get_model(reid_net,  num_classes=751, reid=False):
    return _reid_model[reid_net](num_classes=num_classes, reid=reid)
