from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from feature_extractors.pose_hrnet_backbone import PoseHRNetBackBone
from feature_extractors.swin_unetr_backbone import SwinUNETR_backbone


class ExtractorSelector(object):
    def __init__(self, config, feature_extractor):
        self.config = config
        self.feature_extractor = feature_extractor

    def get_feature_extractor(self, **params):
        model = None
        if "pose_hrnet" == self.feature_extractor:
            model = PoseHRNetBackBone(num_key_points=self.config.segmentation_net_module['NUM_KEY_POINTS'],
                                      num_image_channels=self.config.segmentation_net_module['NUM_IMG_CHANNELS'])
        elif "swin_unetr" == self.feature_extractor:
            model = SwinUNETR_backbone(img_size=self.config.swin_unetr_module['IMG_SIZE'],
                                       in_channels=self.config.swin_unetr_module['IN_CHANNELS'],
                                       out_channels=self.config.swin_unetr_module['OUT_CHANNELS'],
                                       use_checkpoint=self.config.swin_unetr_module['USE_CHECKPOINT'],
                                       spatial_dims=self.config.swin_unetr_module['SPATIAL_DIMS'])

        else:
            print("Feature extractor {} is invalid.".format(self.feature_extractor))
            exit(1)

        return model
