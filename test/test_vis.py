import os
import pdb
import sys
import cv2
import copy
import yaml
import json
import logging
import argparse
from tqdm import tqdm
import os.path as osp
from itertools import groupby
import pycocotools.mask as mask_utils

import numpy as np
import torch
from torchvision.transforms import transforms as T

sys.path[0] = os.getcwd()
from utils.log import logger
from utils.meter import Timer
import data.video as videodataset
from utils import visualize as vis
from utils.mask import temp_interp_mask, mask_seq_jac
from utils.io import write_mot_results, mkdir_if_missing
from core.association import matching
from eval import trackeval

from tracker.mot.mask import MaskAssociationTracker

def fuse_result(res, obs):
    def blank_rle(size):
        brle = np.asfortranarray(np.zeros(size).astype(np.uint8))
        brle = mask_utils.encode(brle)
        brle['counts'] = brle['counts'].decode('ascii')
        return brle 
    size = [o for o in obs[0]['segmentations'] if o is not None][0]['size']
    ret = copy.deepcopy(obs)
    eles = [zip(r[-2], r[-1]) for r in res]
    eles = [(z,t) for t,z in enumerate(eles)]
    eles = [(mask, id_, t) for z,t in eles for (mask, id_) in z]
    idvals = set(map(lambda x:x[1], eles))
    elesbyid = [[(y[0], y[2]) for y in eles if y[1]==x] for x in idvals]
    # mask_seqs: num_objs x seq_len
    mask_seqs = [temp_interp_mask(seq, len(res)) for seq in elesbyid]
    ob_mask_seqs = [o['segmentations'] for o in obs]
    for i, oms in enumerate(ob_mask_seqs):
        for j, it in enumerate(oms):
            if it is None:
                ob_mask_seqs[i][j] = blank_rle(size)
    jac = mask_seq_jac(ob_mask_seqs, mask_seqs)
    #assign_obid = jac.argmax(0)
    matches, u_obs, u_trks = matching.linear_assignment(1-jac, thresh=0.1)
    #pdb.set_trace()
    for i,r in matches:
        ret[i]['segmentations'] = mask_seqs[r]
        #ret.append(ret[i])
        #ret[-1]['segmentations'] = mask_seqs[r]

    return ret


def obs_by_seq(obs):
    ret = dict()
    for j in obs:
        if ret.get(j['video_id'], None):
            ret[j['video_id']].append(j)
        else:
            ret[j['video_id']] = [j,]
    return ret

def obs_by_ins(obs):
    ret = list()
    for x in obs:
        ret.extend(obs[x])
    return ret

def eval_seq(opt, dataloader, save_dir=None):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = MaskAssociationTracker(opt) 
    timer = Timer()
    results = []
    for frame_id, (img, obs, img0, _) in enumerate(dataloader):
        # run tracking
        timer.tic()
        online_targets = tracker.update(img, img0, obs)
        online_tlwhs = []
        online_ids = []
        online_masks = []
        for t in online_targets:
            tlwh = t.tlwh * opt.down_factor
            tid = t.track_id
            mask = t.mask.astype(np.uint8)
            mask = mask_utils.encode(np.asfortranarray(mask))
            mask['counts'] = mask['counts'].decode('ascii')
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_masks.append(mask)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_masks, online_ids))
        if  save_dir is not None:
            online_im = vis.plot_tracking(img0, online_masks, 
                    online_ids, frame_id=frame_id, fps=1. / timer.average_time)
        if save_dir is not None:
            cv2.imwrite(os.path.join(
                save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    return results, timer.average_time, timer.calls


def main(opt):
    logger.setLevel(logging.INFO)
    result_root = opt.out_root 
    mkdir_if_missing(result_root)
    dataroot = osp.join(opt.data_root, opt.split, 'JPEGImages')
    transforms= T.Compose([T.ToTensor(), T.Normalize(opt.im_mean, opt.im_std)])

    obs_file = osp.join(opt.data_root, 'obs', opt.split, opt.obid+'.json')
    meta_file = osp.join(opt.data_root, 'annotations', 'instances_{}_sub.json'.format(opt.split))
    obs = json.load(open(obs_file))
    meta = json.load(open(meta_file))['videos']
    obs = obs_by_seq(obs)
    resobs = dict()
    assert len(obs) == len(meta)

    # run tracking
    accs = []
    timer_avgs, timer_calls = [], []
    for i, seqmeta in enumerate(meta):
        seqobs_all = obs[seqmeta['id']]
        seqobs = [s for s in seqobs_all if s['score']>opt.conf_thres]
        if len(seqobs) < 2: 
            resobs[seqmeta['id']] = [s for s in seqobs_all]
            continue
        seqname = seqmeta['file_names'][0].split('/')[0]
        output_dir = osp.join(result_root, 'frame', seqname)
        dataloader = videodataset.LoadImagesAndMaskObsVIS(dataroot, seqmeta, seqobs, opt)
        seq_res, ta, tc = eval_seq(opt, dataloader, save_dir=output_dir)
        resobs[seqmeta['id']] = fuse_result(seq_res, seqobs) 
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seqname))
        if opt.save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seqname))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

    refined_obs = obs_by_ins(resobs)
    with open(opt.out_file,'w') as f:
        json.dump(refined_obs, f)
    
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))
    
    
    if not opt.split == 'tinytrain':
        return
    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.YouTubeVIS.get_default_dataset_config()
    metrics_config = {'METRICS':['TrackMAP','HOTA','Identity']}

    eval_config['LOG_ON_ERROR'] = osp.join(result_root,'error.log')
    eval_config['PRINT_ONLY_COMBINED'] = True
    dataset_config['GT_FOLDER'] = osp.join(dataroot, '../../annotations/')
    dataset_config['SPLIT_TO_EVAL'] = 'tinytrain'
    dataset_config['TRACKERS_FOLDER'] = osp.join(result_root, '..') 
    dataset_config['TRACKER_SUB_FOLDER'] = '' 
    dataset_config['TRACKERS_TO_EVAL'] = [opt.exp_name, ]
    dataset_config['BENCHMARK'] = 'MOTS20'
    dataset_config['SKIP_SPLIT_FOL'] = True

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.YouTubeVIS(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, 
            trackeval.metrics.Identity, trackeval.metrics.TrackMAP]:
        if metric.get_name() in metrics_config['METRICS']:
            if metric == trackeval.metrics.TrackMAP:
                default_tmap_config = metric.get_default_metric_config()
                default_tmap_config['USE_TIME_RANGES'] = False
                default_tmap_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                                      [128 ** 2, 256 ** 2],
                                                      [256 ** 2, 1e5 ** 2]]
                metrics_list.append(metric(default_tmap_config))
            else:
                metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', required=True, type=str)
    opt = parser.parse_args()
    with open(opt.config) as f:
        common_args = yaml.load(f) 
    for k, v in common_args['common'].items():
        setattr(opt, k, v)
    for k, v in common_args['vis'].items():
        setattr(opt, k, v)
    opt.out_root = osp.join('results/vis', opt.exp_name)
    opt.out_file = osp.join('results/vis', opt.exp_name+'.json')
    print(opt, end='\n\n')

    main(opt)
