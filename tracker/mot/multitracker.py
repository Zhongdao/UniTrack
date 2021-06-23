import os
import pdb
import cv2
import time
import itertools
import os.path as osp
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import ops

from model import AppearanceModel,partial_load
from utils.log import logger
from core.association import matching
from core.propagation import propagate
from core.motion.kalman_filter import KalmanFilter

class AssociationTracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(opt.frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        self.app_model = AppearanceModel(opt).to(opt.device)
        self.app_model.eval()

        def extract_emb(self, img, obs):
            raise NotImplementedError

        def prepare_obs(self, img, img0, obs):
            raise NotImplementedError

        def propagate(self, tracks, img, img0):
            raise NotImplementedError

        def update(self, img, img0, obs):
            raise NotImplementedError


        

