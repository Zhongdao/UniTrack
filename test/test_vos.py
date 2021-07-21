from __future__ import print_function

import os
import sys
sys.path[0] = os.getcwd()
import time
import yaml
import imageio
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from data import vos, jhmdb
from model import AppearanceModel, partial_load
from model.functional import *
import utils
from utils.visualize import dump_predictions


def main(args, vis):
    model = AppearanceModel(args).to(args.device)
    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters())/1000000.0))
    print(model)
    args.mapScale = [args.down_factor, args.down_factor]

    model.eval()
    model = model.to(args.device)

    dataset = vos.VOSDataset(args)
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with torch.no_grad():
        test(val_loader, model, args)

    cvt_path = args.save_path.replace(args.exp_name, 'convert_'+args.exp_name)

    # convert to DAVIS format
    cvt_cmd_str = ('python ./eval/convert_davis.py --in_folder {} '
                   '--out_folder {} --dataset {}').format(
                       args.save_path, cvt_path, args.davisroot)
    eval_cmd_str = ('python {}/evaluation_method.py --task semi-supervised '
                    '--results_path {} --set val --davis_path {}'.format(
                        args.evaltool_root, cvt_path, args.davisroot))
    os.system(cvt_cmd_str)
    os.system(eval_cmd_str)


def test(loader, model, args):
    n_context = args.videoLen
    D = None    # Radius mask

    for vid_idx, (imgs, imgs_orig, lbls, lbls_orig, lbl_map, meta) in enumerate(loader):
        t_vid = time.time()
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        assert(B == 1)

        print('******* Vid %s (%s frames) *******' % (vid_idx, N))
        with torch.no_grad():
            t00 = time.time()

            ##################################################################
            # Compute image features (batched for memory efficiency)
            ##################################################################
            bsize = 5   # minibatch size for computing features
            feats = []
            for b in range(0, imgs.shape[1], bsize):
                feat = model(imgs[:, b:b+bsize].transpose(1, 2).to(args.device))
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2).squeeze(1)

            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            print('computed features', time.time()-t00)

            ##################################################################
            # Compute affinities
            ##################################################################
            torch.cuda.empty_cache()
            t03 = time.time()

            # Prepare source (keys) and target (query) frame features
            key_indices = context_index_bank(n_context, args.long_mem, N - n_context)
            key_indices = torch.cat(key_indices, dim=-1)           
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask TODO use torch.sparse
            restrict = MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            print('computing affinity')
            Ws, Is = mem_efficient_batched_affinity(query, keys, D, 
                        args.temperature, args.topk, args.long_mem, args.device)

            if torch.cuda.is_available():
                print(time.time()-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))

            ##################################################################
            # Propagate Labels and Save Predictions
            ###################################################################

            maps, keypts = [], []
            lbls[0, n_context:] *= 0 
            lbl_map, lbls = lbl_map[0], lbls[0]

            for t in range(key_indices.shape[0]):
                # Soft labels of source nodes
                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                # Weighted sum of top-k neighbours (Is is index, Ws is weight) 
                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])
                pred = pred.permute(1,2,0)
                
                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                # Save Predictions            
                cur_img = imgs_orig[0, t + n_context].permute(1,2,0).numpy() * 255
                _maps = []

                outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))

                heatmap, lblmap, heatmap_prob = dump_predictions(
                    pred.cpu().numpy(),
                    lbl_map, cur_img, outpath)


                _maps += [heatmap, lblmap, heatmap_prob]
                maps.append(_maps)

            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)
            
            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (vid_idx, time.time() - t_vid))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        common_args = yaml.load(f) 
    for k, v in common_args['common'].items():
        setattr(args, k, v)
    for k, v in common_args['vos'].items():
        setattr(args, k, v)

    args.imgSize = args.cropSize
    args.save_path = 'results/vos/{}'.format(args.exp_name)
    args.evaltool_root = osp.join(args.davisroot, 'davis2017-evaluation')

    main(args, None)
