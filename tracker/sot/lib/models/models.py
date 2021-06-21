import math
import torch
import torch.nn as nn
from .ocean import Ocean_
from .oceanplus import OceanPlus_
from .oceanTRT import OceanTRT_
from .cfnet import CFNet
from .siamfc import SiamFC
from .connect import box_tower, AdjustLayer, AlignHead, Corr_Up, MultiDiCorr, OceanCorr
from .backbones import ResNet50, ResNet22W
from .mask import MMS, MSS
from .modules import MultiFeatureBase

import os
import sys

class Ocean(Ocean_):
    def __init__(self, align=False, online=False):
        super(Ocean, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)   # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)
        self.align_head = AlignHead(256, 256) if align else None


class OceanTRT(OceanTRT_):
    def __init__(self, online=False, align=False):
        super(OceanTRT, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)  # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model0 = MultiDiCorr(inchannels=256, outchannels=256)
        self.connect_model1 = box_tower(inchannels=256, outchannels=256, towernum=4)
        self.connect_model2 = OceanCorr()


class OceanPlus(OceanPlus_):
    def __init__(self, online=False, mms=False):
        super(OceanPlus, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)   # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)

        if mms:
            self.mask_model = MMS()
        else:
            self.mask_model = MSS()


