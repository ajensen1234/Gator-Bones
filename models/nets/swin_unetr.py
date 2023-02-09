import torch.nn as nn
from models.feature_extractors.extractor_selector import ExtractorSelector


class Swin_UNETR(nn.Module):
    """
    Currently, this directly calls PoseHighResolutionNetBackBone in feature_extractors/pose_hrnet_modded.py, meaning that
    pose_hrnet should be set as both backbone and head in config.py.
    """

    def __init__(self, config):
        super(Swin_UNETR, self).__init__()
        self.config = config

        if config.model['FEATURE_EXTRACTOR'] != 'swin_unetr':
            print("Swin_unetr head called without pose_hrnet backbone, changing backbone to swin_unetr")
            self.config.model['FEATURE_EXTRACTOR'] = 'swin_unetr'

        self.backbone = ExtractorSelector(config).get_feature_extractor()

    def forward(self, x_):
        x = self.backbone(x_)
        return x
