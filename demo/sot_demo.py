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
import argparse
from PIL import Image
from glob import glob
from os.path import exists, join
from easydict import EasyDict as edict

import torch
import numpy as np

import tracker.sot.lib.models as models
from tracker.sot.lib.utils.utils import  load_dataset, crop_chw, \
    gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from tracker.sot.lib.core.eval_otb import eval_auc_tune

import utils
from model import AppearanceModel, partial_load
from data.vos import color_normalize, load_image, im_to_numpy, im_to_torch


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def preproc(img, im_mean, im_std, use_lab=False):
    img = load_image(img)
    if use_lab:
        img = im_to_numpy(img)
        img = (img*255).astype(np.uint8)[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = im_to_torch(img) / 255.
    img = color_normalize(img, im_mean, im_std)
    if use_lab:
        img = torch.stack([img[0], ]*3)
    img = img.permute(1, 2, 0).numpy()  # H, W, C
    return img


class TrackerConfig(object):
    crop_sz = 512 + 8
    downscale = 8
    temp_sz = crop_sz // downscale

    lambda0 = 1e-4
    padding = 3.5
    interp_factor = 0.01
    num_scale = 3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale // 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.985
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale // 2)))

    net_output_size = [temp_sz, temp_sz]
    cos_window = torch.Tensor(np.outer(np.hanning(temp_sz), np.hanning(temp_sz))).cuda()


def track(net, args):
    toc = 0
    config = TrackerConfig()
    video_name = os.path.basename(args.input) if args.input else 'webcam'
    regions = []  # FINAL RESULTS
    for f, img_raw in enumerate(get_frames(args.input)):
        img_raw = cv2.resize(img_raw, (640,480))
        use_lab = getattr(args, 'use_lab', False)
        im = preproc(img_raw, args.im_mean, args.im_std, use_lab)
        tic = cv2.getTickCount()
        # Init
        if f == 0:
            try:
                init_rect = cv2.selectROI(video_name, img_raw, False, False)
            except Exception:
                exit()
            target_pos, target_sz = rect1_2_cxy_wh(init_rect)
            min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

            # crop template
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(im, bbox, config.crop_sz)

            target = patch
            net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=1)
            regions.append(cxy_wh_2_rect1(target_pos, target_sz))
            patch_crop = np.zeros((config.num_scale, patch.shape[0],
                                   patch.shape[1], patch.shape[2]), np.float32)
        # Track
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

            regions.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

        toc += cv2.getTickCount() - tic
        
        bbox = list(map(int, regions[-1]))
        cv2.rectangle(img_raw, (bbox[0], bbox[1]),
                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
        cv2.imshow(video_name, img_raw)
        cv2.waitKey(40)

    toc /= cv2.getTickFrequency()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', required=True, type=str)
    parser.add_argument('--input', required=True, type=str)
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

    net = models.__dict__[args.arch](base=base, config=TrackerConfig())
    net.eval()
    net = net.cuda()

    track(net, args)


if __name__ == '__main__':
    main()

