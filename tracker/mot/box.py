###################################################################
# File Name: box.py
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
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import ops

from utils.log import logger
from core.association import matching
from core.propagation import propagate
from core.motion.kalman_filter import KalmanFilter

from utils.box import * 
from .basetrack import *
from .multitracker import AssociationTracker

class BoxAssociationTracker(AssociationTracker):
    def __init__(self, opt):
        super(BoxAssociationTracker, self).__init__(opt)

    def extract_emb(self, img, obs):
        feat = self.app_model(img.unsqueeze(0).to(self.opt.device).float())
        scale = [feat.shape[-1]/self.opt.img_size[0], feat.shape[-2]/self.opt.img_size[1]]
        obs_feat = scale_box(scale, obs).to(self.opt.device)
        obs_feat = [obs_feat[:,:4],]
        ret = ops.roi_align(feat, obs_feat, self.opt.feat_size).detach().cpu()
        return ret

    def prepare_obs(self, img, img0, obs):
        ''' Step 1: Network forward, get detections & embeddings'''
        if len(obs) > 0:
            obs = torch.from_numpy(obs[obs[:, 4] > self.opt.conf_thres])
            obs = xywh2xyxy(obs)
            obs = scale_box(self.opt.img_size, obs)
            embs = self.extract_emb(img, obs)
            obs = scale_box_input_size(self.opt.img_size, obs, img0.shape).round()
            '''Detections'''
            detections = [STrack(tlbr_to_tlwh(
                tlbrs[:4]), tlbrs[4], f, self.buffer_size) for (tlbrs, f) in zip(obs, embs)]
        else:
            detections = []
        return detections

    def propagate(self, tracks, img, img0):
        '''
        input: tracks, img
        output: tracks
        '''
        #ret_boxes = propagate(temp_feats, boxes, img, self.app_model, format='box')
        return tracks

    def update(self, img, img0, obs):
        torch.cuda.empty_cache()
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        t1 = time.time()
        detections = self.prepare_obs(img, img0, obs)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        #dists = matching.center_emb_distance(strack_pool, detections)
        dists, recons_ftrk = matching.reconsdot_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections, 
                lambda_=self.opt.motion_lambda, gate=self.opt.motion_gated)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 4: Propagate unassociated tracks, only keep those 
        having high overalps with motion predcitions'''
        if self.opt.prop_flag:
            tracks = [r_tracked_stracks[it] for it in u_track]
            ptracks = self.propagate(tracks, img, img0)
            dists = matching.fuse_motion(self.kalman_filter, 1 - np.eye(len(tracks)), tracks, ptracks)
            remain_track = []
            for i, (t, pt) in enumerate(zip(tracks, ptracks)):
                if dists[i, i] < 0.0:  #NOTE Here
                    t.update(pt, self.frame_id, update_feature=False)
                    activated_stracks.append(t)
                else:
                    remain_track.append(u_track[i])
        else: remain_track = u_track
            

        for it in remain_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_stracks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return output_stracks

