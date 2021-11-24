from __future__ import print_function, absolute_import

import os
import pdb
import os.path as osp
import numpy as np
import math
import cv2
import torch
import time
from matplotlib import cm
from utils import im_to_numpy, im_to_torch


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize(img, (owidth, oheight))
    img = im_to_torch(img)
    return img


def load_image(img):
    # H x W x C => C x H x W
    if isinstance(img, str):
        img = cv2.imread(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = img.copy()
    return im_to_torch(img)


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

######################################################################
def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None

def make_lbl_set(lbls):
    lbl_set = [np.zeros(3).astype(np.uint8)]
    
    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)

    return lbl_set

def texturize(onehot):
    flat_onehot = onehot.reshape(-1, onehot.shape[-1])
    lbl_set = np.unique(flat_onehot, axis=0)

    count_lbls = [np.all(flat_onehot == ll, axis=-1).sum() for ll in lbl_set]
    object_id = np.argsort(count_lbls)[::-1][1]

    hidxs = []
    for h in range(onehot.shape[0]):
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

    return out


class VOSDataset(torch.utils.data.Dataset):
    def __init__(self, args):

        self.davisroot = args.davisroot
        self.split = args.split
        self.imgSize = args.imgSize
        self.videoLen = args.videoLen
        self.mapScale = args.mapScale

        self.texture = False 
        self.round = False 
        self.use_lab = getattr(args, 'use_lab', False)
        self.im_mean = args.im_mean
        self.im_std = args.im_std

        filelist = osp.join(self.davisroot, 'ImageSets/2017', self.split+'.txt')
        f = open(filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            seq = line.strip()

            self.jpgfiles.append(osp.join(self.davisroot,'JPEGImages','480p', seq))
            self.lblfiles.append(osp.join(self.davisroot, 'Annotations','480p', seq))

        f.close()
    
    def get_onehot_lbl(self, lbl_path):
        name = '/' + '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None
    

    def make_paths(self, folder_path, label_path):
        I, L = os.listdir(folder_path), os.listdir(label_path)
        L = [ll for ll in L if 'npy' not in ll]

        frame_num = len(I) + self.videoLen
        I.sort(key=lambda x:int(x.split('.')[0]))
        L.sort(key=lambda x:int(x.split('.')[0]))

        I_out, L_out = [], []

        for i in range(frame_num):
            i = max(0, i - self.videoLen)
            img_path = "%s/%s" % (folder_path, I[i])
            lbl_path = "%s/%s" % (label_path,  L[i])

            I_out.append(img_path)
            L_out.append(lbl_path)

        return I_out, L_out


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        imgs_orig = []
        lbls = []
        lbls_onehot = []
        patches = []
        target_imgs = []

        frame_num = len(os.listdir(folder_path)) + self.videoLen

        img_paths, lbl_paths = self.make_paths(folder_path, label_path)

        t000 = time.time()

        for i in range(frame_num):
            t00 = time.time()

            img_path, lbl_path = img_paths[i], lbl_paths[i]
            img = load_image(img_path)  # CxHxW
            lblimg = cv2.imread(lbl_path)


            '''
            Resize img to 320x320
            '''
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

                lblimg = cv2.resize(lblimg, (newh, neww), cv2.INTER_NEAREST)

            # Resized, but not augmented image
            img_orig = img.clone()
            '''
            Transforms
            '''
            if self.use_lab:
                img = im_to_numpy(img)
                img = (img * 255).astype(np.uint8)[:,:,::-1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img = im_to_torch(img) / 255.
                img = color_normalize(img, self.im_mean, self.im_std)
                img = torch.stack([img[0]]*3)
            else:
                img = color_normalize(img, self.im_mean, self.im_std)

            imgs_orig.append(img_orig)
            imgs.append(img)
            lbls.append(lblimg.copy())
            
        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths)

        ########################################################
        # Load reshaped label information (load cached versions if possible)
        lbls = np.stack(lbls)
        prefix = '/' + '/'.join(lbl_paths[0].split('.')[:-1])

        # Get lblset
        lblset = make_lbl_set(lbls)

        if np.all((lblset[1:] - lblset[:-1]) == 1):
            lblset = lblset[:, 0:1]

        onehots = []
        resizes = []

        rsz_h, rsz_w = math.ceil(img.size(1) / self.mapScale[0]), math.ceil(img.size(2) /self.mapScale[1])

        for i,p in enumerate(lbl_paths):
            prefix = '/' + '/'.join(p.split('.')[:-1])
            # print(prefix)
            oh_path = "%s_%s.npy" % (prefix, 'onehot')
            rz_path = "%s_%s.npy" % (prefix, 'size%sx%s' % (rsz_h, rsz_w))

            onehot = try_np_load(oh_path) 
            if onehot is None:
                print('computing onehot lbl for', oh_path)
                onehot = np.stack([np.all(lbls[i] == ll, axis=-1) for ll in lblset], axis=-1)
                np.save(oh_path, onehot)

            resized = try_np_load(rz_path)
            if resized is None:
                print('computing resized lbl for', rz_path)
                resized = cv2.resize(np.float32(onehot), (rsz_w, rsz_h), cv2.INTER_LINEAR)
                np.save(rz_path, resized)

            if self.texture:
                texturized = texturize(resized)
                resizes.append(texturized)
                lblset = np.array([[0, 0, 0]] + [cm.Paired(i)[:3] for i in range(texturized.shape[-1])]) * 255.0
                break
            else:
                resizes.append(resized)
                onehots.append(onehot)

        if self.texture:
            resizes = resizes * self.videoLen
            for _ in range(len(lbl_paths)-self.videoLen):
                resizes.append(np.zeros(resizes[0].shape))
            onehots = resizes

        ########################################################
        
        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        lbls_tensor = torch.from_numpy(np.stack(lbls))
        lbls_resize = np.stack(resizes)

        assert lbls_resize.shape[0] == len(meta['lbl_paths'])

        return imgs, imgs_orig, lbls_resize, lbls_tensor, lblset, meta

    def __len__(self):
        return len(self.jpgfiles)






