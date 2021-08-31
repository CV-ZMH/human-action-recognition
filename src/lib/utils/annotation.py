from dataclasses import dataclass, field
import numpy as np
from utils.commons import *

@dataclass
class Annotation:
    keypoints : np.ndarray
    bbox : np.ndarray = field(default=None)
    id : int = field(default=None)
    action : list = field(default=None)
    color : str = field(default=None)
    flatten_keypoints: list = field(default=None)

    def set_tracked_id(self, id):
        self.id = id
        self.color = self.set_color_with_id(self.id)

    @staticmethod
    def set_color_with_id(id):
        idx = id % len(COLORS)
        color = list(COLORS.values())[idx]
        return color

    # def add_flatten_skeletons(self):
    #     pass
