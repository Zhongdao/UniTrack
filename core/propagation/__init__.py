###################################################################
# File Name: __init__.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Mon Jan 18 15:57:34 2021
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from .propagate_box import propagate_box
from .propagate_mask import propagate_mask
from .propagate_pose import propagate_pose

def propagate(temp_feats, obs, img, model, format='box'):
    if format == 'box':
        return propagate_box(temp_feats, obs, img, model)
    elif format == 'mask':
        return propagate_box(temp_feats, obs, img, model)
    elif format == 'pose':
        return propagate_pose(temp_feats, obs, img, model)
    else:
        raise ValueError('Observation format not supported.')
