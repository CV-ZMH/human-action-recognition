# -*- coding: utf-8 -*-
from .models.wide_resnet import WideResnet
from .models.siamese_net import SiameseNet
from .models.align_reid import AlignedReid

_reid_model = {
        'wideresnet' : WideResnet,
        'siamesenet' : SiameseNet,
        'align_reid' :  AlignedReid
        }

def get_reid_network(reid_net,  num_classes=751, reid=False):
    return _reid_model[reid_net](num_classes=num_classes, reid=reid)