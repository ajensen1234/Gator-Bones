import torch.nn as nn

from feature_extractors.extractor_selector import ExtractorSelector


class PoseHRNet(nn.Module):
    """
    Currently, this directly calls PoseHighResolutionNet in feature_extractors/pose_hrnet_modded.py, meaning that
    pose_hrnet should be set as both backbone and head in config.py.
    """

    def __init__(self, config):
        super(PoseHRNet, self).__init__()
        self.config = config
        self.backbone = ExtractorSelector(config, "pose_hrnet").get_feature_extractor()

    def forward(self, x_):
        x = self.backbone(x_)
        return x