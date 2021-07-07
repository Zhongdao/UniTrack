import os
import pdb
import os.path as osp
import sys
sys.path[0] = os.getcwd()
import cv2
import copy
import json
import yaml
import logging
import argparse
from tqdm import tqdm
from itertools import groupby
import pycocotools.mask as mask_utils

import numpy as np
import torch
from torchvision.transforms import transforms as T

from utils.log import logger
from utils.meter import Timer
from utils.mask import pts2array
import data.video as videodataset
from utils import visualize as vis
from utils.io import mkdir_if_missing
from core.association import matching

from tracker.mot.pose import PoseAssociationTracker

def identical(a, b):
    if len(a) == len(b):
        arra = pts2array(a)
        arrb = pts2array(b)
        if np.abs(arra-arrb).sum() < 1e-2:
            return True
    return False

def fuse_result(res, jpath):
    with open(jpath, 'r') as f:
        obsj = json.load(f)

    obsj_fused = copy.deepcopy(obsj)
    for t, inpj in enumerate(obsj['annolist']):
        skltns, ids = res[t][2], res[t][3]
        nobj_ori = len(obsj['annolist'][t]['annorect'])
        for i in range(nobj_ori):
            obsj_fused['annolist'][t]['annorect'][i]['track_id'] = [1000]
            for j, skltn in enumerate(skltns):
                match = identical(obsj['annolist'][t]['annorect'][i]['annopoints'][0]['point'], skltn)
                if match:
                    obsj_fused['annolist'][t]['annorect'][i]['track_id'] = [ids[j],]
    return obsj_fused


def eval_seq(opt, dataloader, save_dir=None):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = PoseAssociationTracker(opt) 
    timer = Timer()
    results = []
    for frame_id, (img, obs, img0, _) in enumerate(dataloader):
        # run tracking
        timer.tic()
        online_targets = tracker.update(img, img0, obs)
        online_tlwhs = []
        online_ids = []
        online_poses = []
        for t in online_targets:
            tlwh = t.tlwh 
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_poses.append(t.pose)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_poses, online_ids))
        if  save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, 
                    online_ids, frame_id=frame_id, fps=1. / timer.average_time)
        if save_dir is not None:
            cv2.imwrite(os.path.join(
                save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    return results, timer.average_time, timer.calls


def main(opt):
    logger.setLevel(logging.INFO)
    result_root = opt.out_root 
    result_json_root = osp.join(result_root, 'json')
    mkdir_if_missing(result_json_root)
    transforms= T.Compose([T.ToTensor(), T.Normalize(opt.im_mean, opt.im_std)])

    obs_root = osp.join(opt.data_root, 'obs', opt.split, opt.obid)
    obs_jpaths = [osp.join(obs_root, o) for o in os.listdir(obs_root)]
    obs_jpaths = sorted([o for o in obs_jpaths if o.endswith('.json')])

    # run tracking
    accs = []
    timer_avgs, timer_calls = [], []
    for i, obs_jpath in enumerate(obs_jpaths):
        seqname = obs_jpath.split('/')[-1].split('.')[0]
        output_dir = osp.join(result_root, 'frame', seqname)
        dataloader = videodataset.LoadImagesAndPoseObs(obs_jpath, opt)
        seq_res, ta, tc = eval_seq(opt, dataloader, save_dir=output_dir)
        seq_json = fuse_result(seq_res, obs_jpath) 
        with open(osp.join(result_json_root, "{}.json".format(seqname)), 'w') as f:
            json.dump(seq_json, f)
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seqname))
        if opt.save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seqname))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    cmd_str = ('python ./eval/poseval/evaluate.py --groundTruth={}/posetrack_data/annotations/{} '
               '--predictions={}/ --evalPoseTracking'.format(opt.data_root, opt.split, result_json_root))
    os.system(cmd_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', required=True, type=str)
    opt = parser.parse_args()
    with open(opt.config) as f:
        common_args = yaml.load(f) 
    for k, v in common_args['common'].items():
        setattr(opt, k, v)
    for k, v in common_args['posetrack'].items():
        setattr(opt, k, v)
    
    opt.out_root = osp.join('results/pose', opt.exp_name)
    opt.out_file = osp.join('results/pose', opt.exp_name + '.json')
    print(opt, end='\n\n')

    main(opt)
