import pdb
import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/home/wangzd/datasets/GOT/MOTS/images/train'
label_root = '/home/wangzd/datasets/GOT/MOTS/obs/gt/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = []
    with open(gt_txt, 'r') as f:
        for line in f:
            gt.append(line.strip().split())

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    
    for fid, tid, cid, h, w, m in gt:
        fid = int(fid)
        tid = int(tid)
        cid = int(cid)
        h, w = int(h), int(w)
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '{:d} {:d} {:d} {:d} {:d} {} \n'.format(
                fid, tid, cid, h, w, m)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
