import os
import sys
import pdb
import cv2
import yaml
import logging
import argparse
import os.path as osp
import pycocotools.mask as mask_utils

import numpy as np
import torch
from torchvision.transforms import transforms as T

sys.path[0] = os.getcwd()
from utils.log import logger
from utils.meter import Timer
import data.video as videodataset

from eval import trackeval
from eval.mots.MOTSVisualization import MOTSVisualizer
from utils import visualize as vis
from utils.io import write_mots_results, mkdir_if_missing
from tracker.mot.mask import MaskAssociationTracker

def eval_seq(opt, dataloader, result_filename, save_dir=None, 
        show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    opt.frame_rate = frame_rate
    tracker = MaskAssociationTracker(opt) 
    timer = Timer()
    results = []
    for frame_id, (img, obs, img0, _) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1./max(1e-5, timer.average_time)))

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
            online_im = vis.plot_tracking(img0, online_tlwhs, 
                    online_ids, frame_id=frame_id, fps=1. / timer.average_time)
        if save_dir is not None:
            cv2.imwrite(os.path.join(
                save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

    write_mots_results(result_filename, results)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo', 
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join('results/mots', exp_name, 'quantitive')
    mkdir_if_missing(result_root)


    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join('results/mots', exp_name, 'qualititive', seq) if save_images or save_videos else None
        img_dir = osp.join(data_root, seq, 'img1')
        logger.info('start seq: {}'.format(seq))
        dataloader = videodataset.LoadImagesAndMaskObsMOTS(img_dir, opt)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read() 
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, result_filename,
                save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        if save_videos: 
            visualzier = MOTSVisualizer(seq, None, result_filename, output_dir, img_dir)
            visualzier.generateVideo()

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MOTSChallenge.get_default_dataset_config()
    metrics_config = {'METRICS':['HOTA','CLEAR','Identity']}

    eval_config['LOG_ON_ERROR'] = osp.join(result_root,'error.log')
    eval_config['PLOT_CURVES'] = False
    dataset_config['GT_FOLDER'] = data_root 
    dataset_config['SEQMAP_FOLDER'] = osp.join(data_root, '../../seqmaps')
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['TRACKERS_FOLDER'] = osp.join(result_root, '..') 
    dataset_config['TRACKER_SUB_FOLDER'] = '' 
    dataset_config['TRACKERS_TO_EVAL'] = ['quantitive'] 
    dataset_config['BENCHMARK'] = 'MOTS20'
    dataset_config['SKIP_SPLIT_FOL'] = True

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MOTSChallenge(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, 
            trackeval.metrics.Identity, trackeval.metrics.VACE,
            trackeval.metrics.JAndF]:
        if metric.get_name() in metrics_config['METRICS']:
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
    for k, v in common_args['mots'].items():
        setattr(opt, k, v)    

    print(opt, end='\n\n')

    if not opt.test:
        seqs_str = '''MOTS20-02
                      MOTS20-05
                      MOTS20-09
                      MOTS20-11
                    '''
        data_root = '{}/images/train'.format(opt.mots_root)
    else:
        seqs_str = '''MOTS20-01
                      MOTS20-06
                      MOTS20-07
                      MOTS20-12
                    '''
        data_root = '{}/images/test'.format(opt.mots_root)
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.exp_name,
         show_image=False,
         save_images=opt.save_images, 
         save_videos=opt.save_videos)

