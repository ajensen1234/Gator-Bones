import torch.nn as nn
from feature_extractors.extractor_selector import ExtractorSelector


class Swin_UNETR(nn.Module):
    """
    Name: SWIN_Unetr
    Jiayu Huang

    Inputs:
    x_: input tensor (batch of images)

    Outputs:
    x: output feature tensor from Swin_UNETR feature extractor

    Rationale:
    This is a wrapper for the Swin_UNETR feature extractor, combining Swin Transformer and UNETR architectures. 
    It initializes the backbone and forwards the input through the feature extractor. This class is part of a deep learning pipeline for the segmentation task.

    Future:
    Try to support additional feature extractors, customizing Swin_UNETR architecture, and incorporating new techniques or pre-/post-processing functionalities.
    """
    def __init__(self, config):
        super(Swin_UNETR, self).__init__()
        self.config = config

        if config.model['FEATURE_EXTRACTOR'] != 'swin_unetr':
            print("Swin_unetr head called without pose_hrnet backbone, changing backbone to swin_unetr")
            self.config.model['FEATURE_EXTRACTOR'] = 'swin_unetr'

        self.backbone = ExtractorSelector(config, feature_extractor="swin_unetr").get_feature_extractor()

    def forward(self, x_):
        x = self.backbone(x_)
        return x
