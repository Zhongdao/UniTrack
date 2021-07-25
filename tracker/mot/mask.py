###################################################################
# File Name: mask.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri Jan 29 15:16:53 2021
###################################################################

import numpy as np
import torch
import torch.nn.functional as F

from utils.box import * 
from utils.mask import *
from .basetrack import *
from .multitracker import AssociationTracker

class MaskAssociationTracker(AssociationTracker):
    def __init__(self, opt):
        super(MaskAssociationTracker, self).__init__(opt)

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
                embs.append(torch.randn(d, template_scale))
        return obs, embs

    def prepare_obs(self, img, img0, obs):
        ''' Step 1: Network forward, get detections & embeddings'''
        if obs.shape[0] > 0:
            masks, embs = self.extract_emb(img, obs)
            boxes = mask2box(masks)
            keep_idx = remove_duplicated_box(boxes, iou_th=0.7)
            boxes, masks, obs = boxes[keep_idx], masks[keep_idx], obs[keep_idx]
            embs = [embs[k] for k in keep_idx]
            detections = [STrack(tlbr_to_tlwh(tlbrs), 1, f, self.buffer_size, mask, ac=True) \
                    for (tlbrs,mask,f) in zip(boxes, obs, embs)]
        else:
            detections = []
        return detections

