from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from nets.pose_hrnet import PoseHRNet

SEG_MODEL_DICT = {
    "pose_hrnet": PoseHRNet
}


class ModelManager(object):
    def __init__(self, config):
        self.config = config

    def get_segmentor(self):
        model_name = self.config.model['HEAD']

        if model_name not in SEG_MODEL_DICT:
            print("Model: {} not valid!".format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.config)

        return model
