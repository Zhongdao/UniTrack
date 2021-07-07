import os.path as osp
import os
import numpy as np
import pdb


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

det = 'DET'
seq_root = '/home/wangzd/datasets/MOT/MOT19/images/train'
label_root = '/home/wangzd/datasets/MOT/MOT19/obs/{}/train'.format(det)
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    
    ob_txt = osp.join(seq_root, seq, 'det', 'det.txt')
    gt = np.loadtxt(ob_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, conf,_,_,_  in gt:
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
             x / seq_width, y / seq_height, w / seq_width, h / seq_height, conf)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
