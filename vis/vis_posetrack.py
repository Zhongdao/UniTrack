###################################################################
# File Name: vis_posetrack.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu Mar 25 14:34:58 2021
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import pdb
import cv2
import json
import os.path as osp
import numpy as np

import sys
sys.path[0] = os.getcwd()
from utils.visualize import draw_skeleton, get_color
from utils.mask import pts2array


dataroot = '/home/wangzd/datasets/GOT/Posetrack2018/'
jsonroot = './results/pose/lighttrack_CPN101/json/'
jsonroot = './results/pose/lighttrack_MSRA152/json/'
jsonroot = './results/pose/mocov2_s4/json/'
#jsonroot = '/home/wangzd/datasets/GOT/Posetrack2018/obs/val/lighttrack_CPN101/'
jsonroot = './results/pose/womotion_resnet18_s3/json/'

jsonlist = os.listdir(jsonroot)
jsonlist = [j for j in jsonlist if j.endswith('.json')]
jsonlist = [osp.join(jsonroot, j) for j in jsonlist]

for jp in jsonlist[40:]:
    with open(jp, 'r') as f:
        j_ = json.load(f)['annolist']

    for t, tj in enumerate(j_):
        impath = tj['image'][0]['name']
        impath = osp.join(dataroot, impath)
        im = cv2.imread(impath)
        sklts = tj['annorect']
        if len(sklts) <10:
            break
        for sklt in sklts:
            tid = sklt['track_id'][0]
            pts = sklt['annopoints'][0]['point']
            pts = pts2array(pts)
            c = get_color(tid)
            draw_skeleton(im, pts, c)
            print(tid)
        im = cv2.resize(im, (960,540))
        print('--------------')
        cv2.imshow('f', im)
        cv2.waitKey(10)
        cv2.imwrite('./tmp/{:05d}.jpg'.format(t), im)
            
        
