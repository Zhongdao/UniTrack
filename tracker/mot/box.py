###################################################################
# File Name: box.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri Jan 29 15:16:53 2021
###################################################################

import torch
from torchvision import ops

from .basetrack import STrack
from .multitracker import AssociationTracker
from utils.box import scale_box, scale_box_input_size, xywh2xyxy, tlbr_to_tlwh


class BoxAssociationTracker(AssociationTracker):
    def __init__(self, opt):
        super(BoxAssociationTracker, self).__init__(opt)

    def extract_emb(self, img, obs):
        feat = self.app_model(img.unsqueeze(0).to(self.opt.device).float())
        scale = [feat.shape[-1]/self.opt.img_size[0],
                 feat.shape[-2]/self.opt.img_size[1]]
        obs_feat = scale_box(scale, obs).to(self.opt.device)
        obs_feat = [obs_feat[:, :4], ]
        ret = ops.roi_align(feat, obs_feat, self.opt.feat_size).detach().cpu()
        return ret

    def prepare_obs(self, img, img0, obs):
        if len(obs) > 0:
            obs = torch.from_numpy(obs[obs[:, 4] > self.opt.conf_thres]).float()
            obs = xywh2xyxy(obs)
            obs = scale_box(self.opt.img_size, obs)
            embs = self.extract_emb(img, obs)
            obs = scale_box_input_size(self.opt.img_size, obs, img0.shape)

            if obs.shape[1] == 5:
                detections = [STrack(tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f,
                              self.buffer_size, use_kalman=self.opt.use_kalman)
                              for (tlbrs, f) in zip(obs, embs)]
            elif obs.shape[1] == 6:
                detections = [STrack(tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f,
                              self.buffer_size, category=tlbrs[5],
                              use_kalman=self.opt.use_kalman)
                              for (tlbrs, f) in zip(obs, embs)]
            else:
                raise ValueError(
                        'Shape of observations should be [n, 5] or [n, 6].')
        else:
            detections = []
        return detections
