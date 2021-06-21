###################################################################
# File Name: functional.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Mon Jun 21 21:04:09 2021
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_prop(pred):
    pred_max = pred.max(axis=0)[0]
    pred[pred <  pred_max] = 0
    pred[pred >= pred_max] = 1
    pred /= pred.sum(0)[None]
    return pred

def context_index_bank(n_context, long_mem, N):
    '''
    Construct bank of source frames indices, for each target frame
    '''
    ll = []   # "long term" context (i.e. first frame)
    for t in long_mem:
        assert 0 <= t < N, 'context frame out of bounds'
        idx = torch.zeros(N, 1).long()
        if t > 0:
            idx += t + (n_context+1)
            idx[:n_context+t+1] = 0
        ll.append(idx)
    # "short" context    
    ss = [(torch.arange(n_context)[None].repeat(N, 1) +  \
            torch.arange(N)[:, None])[:, :]]
    return ll + ss


def mem_efficient_batched_affinity(
        query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    '''
    bsize, pbsize = 10, 100 #keys.shape[2] // 2
    Ws, Is = [], []

    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        for pb in range(0, _k.shape[-1], pbsize):
            A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
            A[0, :, len(long_mem):] += mask[..., pb:pb+pbsize].to(device)

            _, N, T, h1w1, hw = A.shape
            A = A.view(N, T*h1w1, hw)
            A /= temperature

            weights, ids = torch.topk(A, topk, dim=-2)
            weights = F.softmax(weights, dim=-2)
            
            w_s.append(weights.cpu())
            i_s.append(ids.cpu())

        weights = torch.cat(w_s, dim=-1)
        ids = torch.cat(i_s, dim=-1)
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    return Ws, Is

def batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    (less aggressively mini-batched)
    '''
    bsize = 2
    Ws, Is = [], []
    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        A = torch.einsum('ijklmn,ijkop->iklmnop', _k, _q) / temperature
        
        # Mask
        A[0, :, len(long_mem):] += mask.to(device)

        _, N, T, h1w1, hw = A.shape
        A = A.view(N, T*h1w1, hw)
        A /= temperature

        weights, ids = torch.topk(A, topk, dim=-2)
        weights = F.softmax(weights, dim=-2)
            
        Ws += [w for w in weights]
        Is += [ii for ii in ids]
    
    return Ws, Is

def process_pose(pred, lbl_set, topk=3):
    # generate the coordinates:
    pred = pred[..., 1:]
    flatlbls = pred.flatten(0,1)
    topk = min(flatlbls.shape[0], topk)
    
    vals, ids = torch.topk(flatlbls, k=topk, dim=0)
    vals /= vals.sum(0)[None]
    xx, yy = ids % pred.shape[1], ids // pred.shape[1]

    current_coord = torch.stack([(xx * vals).sum(0), (yy * vals).sum(0)], dim=0)
    current_coord[:, flatlbls.sum(0) == 0] = -1

    pred_val_sharp = np.zeros((*pred.shape[:2], 3))

    for t in range(len(lbl_set) - 1):
        x = int(current_coord[0, t])
        y = int(current_coord[1, t])

        if x >=0 and y >= 0:
            pred_val_sharp[y, x, :] = lbl_set[t + 1]

    return current_coord.cpu(), pred_val_sharp

class MaskedAttention(nn.Module):
    '''
    A module that implements masked attention based on spatial locality 
    TODO implement in a more efficient way (torch sparse or correlation filter)
    '''
    def __init__(self, radius, flat=True):
        super(MaskedAttention, self).__init__()
        self.radius = radius
        self.flat = flat
        self.masks = {}
        self.index = {}

    def mask(self, H, W):
        if not ('%s-%s' %(H,W) in self.masks):
            self.make(H, W)
        return self.masks['%s-%s' %(H,W)]

    def index(self, H, W):
        if not ('%s-%s' %(H,W) in self.index):
            self.make_index(H, W)
        return self.index['%s-%s' %(H,W)]

    def make(self, H, W):
        if self.flat:
            H = int(H**0.5)
            W = int(W**0.5)
        
        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ( (gx[None, None, :, :] - gx[:, :, None, None])**2 + (gy[None, None, :, :] - gy[:, :, None, None])**2 ).float() ** 0.5
        D = (D < self.radius)[None].float()

        if self.flat:
            D = self.flatten(D)
        self.masks['%s-%s' %(H,W)] = D

        return D

    def flatten(self, D):
        return torch.flatten(torch.flatten(D, 1, 2), -2, -1)

    def make_index(self, H, W, pad=False):
        mask = self.mask(H, W).view(1, -1).byte()
        idx = torch.arange(0, mask.numel())[mask[0]][None]

        self.index['%s-%s' %(H,W)] = idx

        return idx
        
    def forward(self, x):
        H, W = x.shape[-2:]
        sid = '%s-%s' % (H,W)
        if sid not in self.masks:
            self.masks[sid] = self.make(H, W).to(x.device)
        mask = self.masks[sid]
        return x * mask[0]
