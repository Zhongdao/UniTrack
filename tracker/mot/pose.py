###################################################################
# File Name: mask.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri Jan 29 15:16:53 2021
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import pdb
import cv2
import time
import itertools
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import ops

from utils.box import * 
from utils.mask import *
from .basetrack import *
from .multitracker import AssociationTracker

class PoseAssociationTracker(AssociationTracker):
    def __init__(self, opt):
        super(PoseAssociationTracker, self).__init__(opt)

    def extract_emb(self, img, obs):
        img = img.unsqueeze(0).to(self.opt.device).float()
        with torch.no_grad():
            feat = self.app_model(img)
        _, d, h, w = feat.shape
        obs = torch.from_numpy(obs).to(self.opt.device).float()
        obs = F.interpolate(obs.unsqueeze(1), size=(h,w), mode='nearest')
        template_scale = np.prod(self.opt.feat_size)
        embs = []
        for ob in obs:
            obfeat = ob*feat
            scale = ob.sum()
            if scale > 0:
                if scale > self.opt.max_mask_area:
                    scale_factor = np.sqrt(self.opt.max_mask_area/scale.item())
                else:
                    scale_factor = 1
                norm_obfeat = F.interpolate(obfeat, scale_factor=scale_factor, mode='bilinear')
                norm_mask = F.interpolate(ob.unsqueeze(1), scale_factor=scale_factor, mode='nearest')
                emb = norm_obfeat[:,:, norm_mask.squeeze(0).squeeze(0).ge(0.5)]
                embs.append(emb.cpu())
            else: 
                embs.append(torch.randn(1, d, template_scale).cpu())
        return obs.cpu(), embs

    def prepare_obs(self, img, img0, obs):
        _, h, w = img.shape
        ''' Step 1: Network forward, get detections & embeddings'''
        if len(obs) > 0:
            masks = list()
            for ob in obs:
                mask = skltn2mask(ob, (h,w))
                masks.append(mask)
            masks = np.stack(masks)
            masks, embs = self.extract_emb(img, masks)
            boxes = [skltn2box(ob) for ob in obs]
            assert len(obs)==len(boxes)
            detections = [STrack(tlbr_to_tlwh(tlbrs), 1, f, self.buffer_size, mask, pose, ac=True) \
                    for (tlbrs,mask,pose,f) in zip(boxes,masks,obs,embs)]
        else:
            detections = []
        return detections

