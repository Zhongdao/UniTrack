from collections import defaultdict, deque
import datetime
import time
import torch

import errno
import os
import pdb
import sys

from . import arguments
from . import visualize
from . import box
from . import meter
from . import log


from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import model.resnet as resnet
import model.hrnet as hrnet
import model.random_feat_generator as random_feat_generator

#########################################################
# DEBUG
#########################################################

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

sys.excepthook = info



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


#################################################################################
### Network Utils
#################################################################################

def partial_load(pretrained_dict, model, skip_keys=[], log=True):
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
    if model_type == 'scratch':
        net = resnet.resnet18()
        if args.nopadding:
            net.modify(padding='no')
        else:
            net.modify(padding='reflect')

    elif model_type == 'scratch_zeropad':
        net = resnet.resnet18()

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

#################################################################################
### Misc
#################################################################################

def sinkhorn_knopp(A, tol=0.01, max_iter=1000, verbose=False):
    _iter = 0
    
    if A.ndim > 2:
        A = A / A.sum(-1).sum(-1)[:, None, None]
    else:
        A = A / A.sum(-1).sum(-1)[None, None]

    A1 = A2 = A 

    while (A2.sum(-2).std() > tol and _iter < max_iter) or _iter == 0:
        A1 = F.normalize(A2, p=1, dim=-2)
        A2 = F.normalize(A1, p=1, dim=-1)

        _iter += 1
        if verbose:
            print(A2.max(), A2.min())
            print('row/col sums', A2.sum(-1).std().item(), A2.sum(-2).std().item())

    if verbose:
        print('------------row/col sums aft', A2.sum(-1).std().item(), A2.sum(-2).std().item())

    return A2 

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img
