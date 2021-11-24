import pdb
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np

import utils
import model.resnet as resnet
import model.hrnet as hrnet
import model.random_feat_generator as random_feat_generator

class AppearanceModel(nn.Module):
    def __init__(self, args):
        super(AppearanceModel, self).__init__()
        self.args = args
        
        self.model = make_encoder(args).to(self.args.device)
    def forward(self, x):
        z = self.model(x)
        return z

def partial_load(pretrained_dict, model, skip_keys=[], log=False):
    model_dict = model.state_dict()
    
    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not any([sk in k for sk in skip_keys])}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]
    unload_keys = [k for k in model_dict if k not in pretrained_dict]
    
    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if log:
        print('\nSkipped keys: ', skipped_keys)
        print('\nLoading keys: ', filtered_dict.keys())
        print('\nUnLoaded keys: ', unload_keys)

def load_vince_model(path):
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    checkpoint = {k.replace('feature_extractor.module.model.', ''): checkpoint[k] for k in checkpoint if 'feature_extractor' in k}
    return checkpoint


def load_uvc_model(ckpt_path):
    net = resnet.resnet18()
    net.avgpool, net.fc = None, None

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = {k.replace('module.gray_encoder.', ''):v for k,v in ckpt['state_dict'].items() if 'gray_encoder' in k}
    partial_load(state_dict, net)

    return net


def load_tc_model(ckpt_path):
    model_state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    
    net = resnet.resnet50()
    net_state = net.state_dict()

    for k in [k for k in model_state.keys() if 'encoderVideo' in k]:
        kk = k.replace('module.encoderVideo.', '')
        tmp = model_state[k]
        if net_state[kk].shape != model_state[k].shape and net_state[kk].dim() == 4 and model_state[k].dim() == 5:
            tmp = model_state[k].squeeze(2)
        net_state[kk][:] = tmp[:]
        
    net.load_state_dict(net_state)

    return net

class From3D(nn.Module):
    ''' Use a 2D convnet as a 3D convnet '''
    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet
    
    def forward(self, x):
        N, C, T, h, w = x.shape
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w)
        m = self.model(xx)

        return m.view(N, T, *m.shape[-3:]).permute(0, 2, 1, 3, 4)


def make_encoder(args):
    SSL_MODELS = ['byol', 'deepcluster-v2', 'infomin', 'insdis', 'moco-v1', 'moco-v2',
            'pcl-v1', 'pcl-v2','pirl', 'sela-v2', 'swav', 'simclr-v1', 'simclr-v2',
            'pixpro', 'detco', 'barlowtwins']
    model_type = args.model_type
    if model_type == 'crw':
        net = resnet.resnet18()
        if osp.isfile(args.resume):
            ckpt = torch.load(args.resume)
            state = {}
            for k, v in ckpt['model'].items():
                if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                    state[k.replace('.1.weight', '.weight')] = v
                if 'encoder.model' in k:
                    state[k.replace('encoder.model.', '')] = v
                else:
                    state[k] = v
            partial_load(state, net, skip_keys=['head',])
            del ckpt
    elif model_type == 'random18':
        net = resnet.resnet18(pretrained=False)
    elif model_type == 'random50': 
        net = resnet.resnet50(pretrained=False)
    elif model_type == 'imagenet18':
        net = resnet.resnet18(pretrained=True)
    elif model_type == 'imagenet50':
        net = resnet.resnet50(pretrained=True)
    elif model_type == 'imagenet101':
        net = resnet.resnet101(pretrained=True)
    elif model_type == 'imagenet_resnext50':
        net = resnet.resnext50_32x4d(pretrained=True)
    elif model_type == 'imagenet_resnext101':
        net = resnet.resnext101_32x8d(pretrained=True)
    elif model_type == 'mocov2':
        net = resnet.resnet50(pretrained=False)
        net_ckpt = torch.load(args.resume)
        net_state = {k.replace('module.encoder_q.', ''):v for k,v in net_ckpt['state_dict'].items() \
                if 'module.encoder_q' in k}
        partial_load(net_state, net)
    elif model_type == 'ssib':
        net = resnet.resnet50(pretrained=False)
        net_ckpt = torch.load(args.resume)
        net_state = {k.replace('module.encoder.', ''):v for k,v in net_ckpt.items() \
                if 'module.encoder' in k}
        partial_load(net_state, net)
    elif model_type == 'uvc':
        net = load_uvc_model(args.resume)
    elif model_type == 'timecycle':
        net = load_tc_model(args.resume)
    elif model_type in SSL_MODELS:
        net = resnet.resnet50(pretrained=False)
        net_ckpt = torch.load(args.resume)
        partial_load(net_ckpt, net)
    elif 'hrnet' in model_type:
        net = hrnet.get_cls_net(model_type, return_stage=args.return_stage, pretrained=args.resume)
    elif model_type == 'random':
        net = random_feat_generator.RandomFeatGenerator(args)
    else:
        raise ValueError('Invalid model_type.')
    if hasattr(net, 'modify'):
        net.modify(remove_layers=args.remove_layers)

    if 'Conv2d' in str(net) and not args.infer2D:
        net = From3D(net)
    return net
