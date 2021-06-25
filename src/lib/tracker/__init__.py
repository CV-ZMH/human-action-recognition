# -*- coding: utf-8 -*-
from .deepsort.deepsort import DeepSort
# from .norfair.norfair import NorFair

trackers = {
    'deepsort' : DeepSort,
    # 'norfair' : NorFair #TODO add implementation on master branch
}

def get_tracker(name, **kwargs):
    return trackers[name](**kwargs)
