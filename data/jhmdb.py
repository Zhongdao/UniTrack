from __future__ import print_function, absolute_import

import os
import numpy as np
import math
import scipy.io as sio

import cv2
import torch
from matplotlib import cm

from utils import im_to_numpy, im_to_torch

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize( img, (owidth, oheight) )
    img = im_to_torch(img)
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
#     print(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:,:,::-1]
    img = img.copy()
    return im_to_torch(img)

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

import time




######################################################################
def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None

def make_lbl_set(lbls):
    print(lbls.shape)
    t00 = time.time()

    lbl_set = [np.zeros(3).astype(np.uint8)]
    count_lbls = [0]    
    
    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)

    # print(lbl_set)
    # if (lbl_set > 20).sum() > 0:
    #     import pdb; pdb.set_trace()
    # count_lbls = [np.all(flat_lbls_0 == ll, axis=-1).sum() for ll in lbl_set]
    
    print('lbls', time.time() - t00)

    return lbl_set


def texturize(onehot):
    flat_onehot = onehot.reshape(-1, onehot.shape[-1])
    lbl_set = np.unique(flat_onehot, axis=0)

    count_lbls = [np.all(flat_onehot == ll, axis=-1).sum() for ll in lbl_set]
    object_id = np.argsort(count_lbls)[::-1][1]

    hidxs = []
    for h in range(onehot.shape[0]):
        # appears = any(np.all(onehot[h] == lbl_set[object_id], axis=-1))
        appears = np.any(onehot[h, :, 1:] == 1)
        if appears:    
            hidxs.append(h)

    nstripes = min(10, len(hidxs))

    out = np.zeros((*onehot.shape[:2], nstripes+1))
    out[:, :, 0] = 1

    for i, h in enumerate(hidxs):
        cidx = int(i // (len(hidxs) / nstripes))
        w = np.any(onehot[h, :, 1:] == 1, axis=-1)
        out[h][w] = 0
        out[h][w, cidx+1] = 1
        # print(i, h, cidx)

    return out



class JhmdbSet(torch.utils.data.Dataset):
    def __init__(self, args, sigma=0.5):

        self.filelist = args.filelist
        self.imgSize = args.imgSize
        self.videoLen = args.videoLen
        self.mapScale = args.mapScale

        self.sigma = sigma

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = rows[1]
            lblfile = rows[0]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()
    
    def get_onehot_lbl(self, lbl_path):
        name = '/' + '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None
    

    def make_paths(self, folder_path, label_path):
        I = [ ll for ll in os.listdir(folder_path) if '.png' in ll ]

        frame_num = len(I) + self.videoLen
        I.sort(key=lambda x:int(x.split('.')[0]))

        I_out, L_out = [], []

        for i in range(frame_num):
            i = max(0, i - self.videoLen)
            img_path = "%s/%s" % (folder_path, I[i])

            I_out.append(img_path)

        return I_out


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        imgs_orig = []
        lbls = []
        lbls_onehot = []
        patches = []
        target_imgs = []
        
        img_paths = self.make_paths(folder_path, label_path)
        frame_num = len(img_paths)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        t000 = time.time()

        # frame_num = 30
        for i in range(frame_num):
            t00 = time.time()

            img_path = img_paths[i]
            img = load_image(img_path)  # CxHxW

            # print('loaded', i, time.time() - t00)

            ht, wd = img.size(1), img.size(2)
            if self.imgSize > 0:
                newh, neww = ht, wd

                if ht <= wd:
                    ratio  = 1.0 #float(wd) / float(ht)
                    # width, height
                    img = resize(img, int(self.imgSize * ratio), self.imgSize)
                    newh = self.imgSize
                    neww = int(self.imgSize * ratio)
                else:
                    ratio  = 1.0 #float(ht) / float(wd)
                    # width, height
                    img = resize(img, self.imgSize, int(self.imgSize * ratio))
                    newh = int(self.imgSize * ratio)
                    neww = self.imgSize


            img_orig = img.clone()
            img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)

        rsz_h, rsz_w = math.ceil(img.size(1) / self.mapScale[0]), math.ceil(img.size(2) /self.mapScale[1])

        lbls_mat = sio.loadmat(label_path)

        lbls_coord = lbls_mat['pos_img']
        lbls_coord = lbls_coord - 1


        lbls_coord[0, :, :] = lbls_coord[0, :, :] * float(neww) / float(wd) / self.mapScale[0]
        lbls_coord[1, :, :] = lbls_coord[1, :, :] * float(newh) / float(ht) / self.mapScale[1]
        lblsize =  (rsz_h, rsz_w)

        lbls = np.zeros((lbls_coord.shape[2], lblsize[0], lblsize[1], lbls_coord.shape[1]))

        for i in range(lbls_coord.shape[2]):
            lbls_coord_now = lbls_coord[:, :, i]
            scales = lbls_coord_now.max(1) - lbls_coord_now.min(1)
            scale = scales.max()
            scale = max(0.5, scale*0.015)
            for j in range(lbls_coord.shape[1]):
                if self.sigma > 0:
                    draw_labelmap_np(lbls[i, :, :, j], lbls_coord_now[:, j], scale)
                else:
                    tx = int(lbls_coord_now[0, j])
                    ty = int(lbls_coord_now[1, j])
                    if tx < lblsize[1] and ty < lblsize[0] and tx >=0 and ty >=0:
                        lbls[i, ty, tx, j] = 1.0

        lbls_tensor = torch.zeros(frame_num, lblsize[0], lblsize[1], lbls_coord.shape[1])

        for i in range(frame_num):
            if i < self.videoLen:
                nowlbl = lbls[0]
            else:
                if(i - self.videoLen < len(lbls)):
                    nowlbl = lbls[i - self.videoLen]
            lbls_tensor[i] = torch.from_numpy(nowlbl)
        
        lbls_tensor = torch.cat([(lbls_tensor.sum(-1) == 0)[..., None] *1.0, lbls_tensor], dim=-1)

        lblset = np.arange(lbls_tensor.shape[-1]-1)
        lblset = np.array([[0, 0, 0]] + [cm.Paired(i)[:3] for i in lblset]) * 255.0

        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=[])
        
        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        lbls_resize = lbls_tensor #np.stack(resizes)

        assert lbls_resize.shape[0] == len(meta['img_paths'])
        #print('vid', i, 'took', time.time() - t000)

        return imgs, imgs_orig, lbls_resize, lbls_tensor, lblset, meta

    def __len__(self):
        return len(self.jpgfiles)


def draw_labelmap_np(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img
