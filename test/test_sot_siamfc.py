# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------

import os
import pdb
import sys
sys.path[0] = os.getcwd()
import cv2
import yaml
import random
import argparse
from os.path import exists, join
from easydict import EasyDict as edict

import torch
import numpy as np

import tracker.sot.lib.models.models as models
from tracker.sot.lib.utils.utils import  load_dataset, crop_chw, \
    gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from tracker.sot.lib.core.eval_otb import eval_auc_tune

import utils
from model import AppearanceModel, partial_load
from data.vos import color_normalize, load_image, im_to_numpy, im_to_torch

def sot_loadimg(path, im_mean, im_std, use_lab=False):
    img = load_image(path)
    if use_lab:
        img = im_to_numpy(img)
        img = (img*255).astype(np.uint8)[:,:,::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = im_to_torch(img) / 255.
    img = color_normalize(img, im_mean, im_std)
    if use_lab:
        img = torch.stack([img[0],]*3)
    img = img.permute(1,2,0).numpy()  # H, W, C
    return img 

class TrackerConfig(object):
    crop_sz = 512 + 8 
    downscale = 8
    temp_sz = crop_sz // downscale

    lambda0 = 1e-4
    padding = 3.5
    interp_factor = 0.01
    num_scale =  3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale // 2)
    min_scale_factor = 0.2
    max_scale_factor = 5 
    scale_penalty = 0.985  
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale // 2)))

    net_output_size = [temp_sz, temp_sz]
    cos_window = torch.Tensor(np.outer(np.hanning(temp_sz), np.hanning(temp_sz))).cuda()


def track(net, video, args):
    start_frame, toc = 0, 0
    config = TrackerConfig()
    # save result to evaluate
    if args.exp_name:
        tracker_path = join('results', 'sot', args.arch, args.exp_name)
    else:
        tracker_path = join('results', 'sot', args.arch, 'unknown')

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting

    regions = [] # FINAL RESULTS
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        use_lab = getattr(args, 'use_lab', False)
        im = sot_loadimg(image_file, args.im_mean, args.im_std, use_lab)
        tic = cv2.getTickCount()
	### Init
        if f == 0:
            target_pos, target_sz = rect1_2_cxy_wh(gt[0])
            min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

            # crop template
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(im, bbox, config.crop_sz)

            target = patch 
            net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=1)
            regions.append(cxy_wh_2_rect1(target_pos, target_sz))
            patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
        ### Track
        else:
            for i in range(config.num_scale):  # crop multi-scale search region
                window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
                bbox = cxy_wh_2_bbox(target_pos, window_sz)
                patch_crop[i, :] = crop_chw(im, bbox, config.crop_sz)

            search = patch_crop 
            response = net(torch.Tensor(search).cuda())
            net_output_size = [response.shape[-2], response.shape[-1]]
            peak, idx = torch.max(response.view(config.num_scale, -1), 1)
            peak = peak.data.cpu().numpy() * config.scale_penalties
            best_scale = np.argmax(peak)
            r_max, c_max = np.unravel_index(idx[best_scale].cpu(), net_output_size)

            r_max = r_max - net_output_size[0] * 0.5
            c_max = c_max - net_output_size[1] * 0.5
            window_sz = target_sz * (config.scale_factor[best_scale] * (1 + config.padding))

            target_pos = target_pos + np.array([c_max, r_max]) * window_sz / net_output_size 
            target_sz = np.minimum(np.maximum(window_sz / (1 + config.padding), min_sz), max_sz)

            # model update
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(im, bbox, config.crop_sz)
            target = patch 
            #net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=config.interp_factor)

            regions.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

        toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        else:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        common_args = yaml.load(f) 
    for k, v in common_args['common'].items():
        setattr(args, k, v)
    for k, v in common_args['sot'].items():
        setattr(args, k, v)
    args.arch = 'SiamFC'

    # prepare model
    base = AppearanceModel(args).to(args.device) 
    print('Total params: %.2fM' % 
            (sum(p.numel() for p in base.parameters())/1e6))
    print(base)
    if os.path.isfile(args.resume):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume) 
        if args.model_type == 'crw' or args.model_type=='imagenet18':
            state = {}
            for k,v in checkpoint['model'].items():
                if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                    state[k.replace('.1.weight', '.weight')] = v
                elif 'encoder.model' in k:
                    state[k.replace('encoder.model', 'model')] = v
                else:
                    state[k] = v
            partial_load(state, base, skip_keys=['head'])
        del checkpoint
    
    net = models.__dict__[args.arch](base=base, config=TrackerConfig())
    net.eval()
    net = net.cuda()

    # prepare video
    dataset = load_dataset(args.dataset, args.dataroot)
    video_keys = list(dataset.keys()).copy()

    # tracking all videos in benchmark
    for video in video_keys:
        track(net, dataset[video], args)

    eval_cmd = ('python ./tracker/sot/lib/eval_toolkit/bin/eval.py --dataset_dir {} '
                '--tracker_result_dir ./results/sot/{} --trackers {} --dataset OTB2015').format(
                args.dataroot, args.arch, args.exp_name)
    os.system(eval_cmd)

if __name__ == '__main__':
    main()

