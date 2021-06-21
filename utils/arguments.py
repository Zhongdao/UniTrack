import argparse
import os
import torch
import random
import utils

def common_args(parser):
    return parser

def test_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Label Propagation')

    # Datasets
    parser.add_argument('--exp-name', default='tmp', type=str)
    parser.add_argument('--nopadding', default=False, action='store_true')
    parser.add_argument('--infer2D', default=False, action='store_true')
    parser.add_argument('--dataset', default='OTB2015', type=str)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--manualSeed', type=int, default=777, help='manual seed')
    # video walk
    parser.add_argument('--im-mean', default=[0.4914, 0.4822, 0.4465], 
            type=float, nargs='*', help='')
    parser.add_argument('--im-std', default=[0.2023, 0.1994, 0.2010], 
            type=float, nargs='*', help='')
    # pytorch pretrained
    #parser.add_argument('--im-mean', default=[0.485, 0.456, 0.406], 
    #        type=float, nargs='*', help='')
    #parser.add_argument('--im-std', default=[0.229, 0.224, 0.225], 
    #        type=float, nargs='*', help='')
    parser.add_argument('--feat-size', default=[4,10], type=int, nargs='*', help='')

    #Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batchSize', default=1, type=int,
                        help='batchSize')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='temperature')
    parser.add_argument('--topk', default=10, type=int,
                        help='k for kNN')
    parser.add_argument('--radius', default=12, type=float,
                        help='spatial radius to consider neighbors from')
    parser.add_argument('--videoLen', default=20, type=int,
                        help='number of context frames')

    parser.add_argument('--cropSize', default=320, type=int,
                        help='resizing of test image, -1 for native size')

    parser.add_argument('--filelist', default='/scratch/ajabri/data/davis/val2017.txt', type=str)
    parser.add_argument('--save-path', default='./results', type=str)

    parser.add_argument('--visdom', default=False, action='store_true')
    parser.add_argument('--visdom-server', default='localhost', type=str)

    # Model Details
    parser.add_argument('--model-type', default='scratch', type=str)
    parser.add_argument('--head-depth', default=-1, type=int,
                        help='depth of mlp applied after encoder (0 = linear)')

    parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')
    #parser.add_argument('--remove-layers', default=[], help='layer[1-4]')
    parser.add_argument('--no-l2', default=False, action='store_true', help='')

    parser.add_argument('--long-mem', default=[0], type=int, nargs='*', help='')
    parser.add_argument('--texture', default=False, action='store_true', help='')
    parser.add_argument('--round', default=False, action='store_true', help='')

    parser.add_argument('--norm_mask', default=False, action='store_true', help='')
    parser.add_argument('--finetune', default=0, type=int, help='')
    parser.add_argument('--pca-vis', default=False, action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using GPU', args.gpu_id)
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # Set seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    return args

