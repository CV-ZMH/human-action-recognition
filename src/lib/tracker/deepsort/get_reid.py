# -*- coding: utf-8 -*-
from .models.wide_resnet import WideResnet
from .models.siamese_net import SiameseNet
from .models.osnet import osnet_ibn_x1_0
from .models.mudeep import MuDeep


_reid_model = {
        'wideresnet' : WideResnet,
        'siamesenet' : SiameseNet,
        'osnet_ibn_x1_0': osnet_ibn_x1_0
        }

def get_reid_network(reid_name, num_classes=751, reid=False):
    avai_models = list(_reid_model.keys())
    if reid_name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(
            reid_name, avai_models))
    return _reid_model[reid_name](num_classes=num_classes, reid=reid)
