from .market import Market1501
from .mars import Mars
from .siamese_dataset import SiameseTriplet
from .utils import *

datasets = {
    'market': Market1501,
    'mars': Mars
    }

def get_dataset(reid_net, dataset_name, **kwargs):
    if reid_net == 'siamese':
        return SiameseTriplet(**kwargs)

    return datasets[dataset_name](**kwargs)
