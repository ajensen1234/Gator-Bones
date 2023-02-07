from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from feature_extractors.pose_hrnet_modded_in_notebook import PoseHighResolutionNet

class ExtractorSelector(object):
    def __init__(self, config):
        self.config = config

    def get_feature_extractor(self, **params):
        feature_extractor = self.config.model["FEATURE_EXTRACTOR"]

        model = None
        if "pose_hrnet" in feature_extractor:
            model = PoseHighResolutionNet(num_key_points=self.config.segmentation_net_module['NUM_KEY_POINTS'],
                                  num_image_channels=self.config.segmentation_net_module['NUM_IMG_CHANNELS'])

        else:
            print("Feature extractor {} is invalid.".format(feature_extractor))
            exit(1)

        return model
