import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import utils

EPS = 1e-20


class CRW(nn.Module):
    def __init__(self, args, vis=None):
        super(CRW, self).__init__()
        self.args = args

        self.edgedrop_rate = getattr(args, 'dropout', 0)
        self.featdrop_rate = getattr(args, 'featdrop', 0)
        self.temperature = getattr(args, 'temp', getattr(args, 'temperature', 0.07))

        self.encoder = utils.make_encoder(args).to(self.args.device)
        self.infer_dims()
        self.selfsim_fc = self.make_head(depth=getattr(args, 'head_depth', 0))

        self.xent = nn.CrossEntropyLoss(reduction="none")
        self._xent_targets = dict()

        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)
        self.featdrop = nn.Dropout(p=self.featdrop_rate, inplace=False)

        self.flip = getattr(args, 'flip', False)
        self.sk_targets = getattr(args, 'sk_targets', False)
        self.vis = vis

    def infer_dims(self):
        self.encoder.eval()
        in_sz = 256
        dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        if self.args.infer2D:
            dummy = dummy.squeeze(2)
        dummy_out = self.encoder(dummy)
        self.enc_hid_dim = dummy_out.shape[1]
        self.map_scale = in_sz // dummy_out.shape[-1]

    def make_head(self, depth=1):
        head = []
        if depth >= 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [128]
            for d1, d2 in zip(dims, dims[1:]):
                h = nn.Linear(d1, d2)
                head += [h, nn.ReLU()]
            head = head[:-1]

        return nn.Sequential(*head)

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1).float().cuda()
        return A * mask

    def affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

        A = torch.einsum('bctn,bctm->btnm', x1, x2)
        # if self.restrict is not None:
        #     A = self.restrict(A)

        return A.squeeze(1) if in_t_dim < 4 else A
    
    def stoch_mat(self, A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False):
        ''' Affinity -> Stochastic Matrix '''

        if zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.edgedrop_rate > 0:
            A[torch.rand_like(A) < self.edgedrop_rate] = -1e20

        if do_sinkhorn:
            return utils.sinkhorn_knopp((A/self.temperature).exp(), 
                tol=0.01, max_iter=100, verbose=False)

        return F.softmax(A/self.temperature, dim=-1)

    def pixels_to_nodes(self, x):
        ''' 
            pixel maps -> node embeddings 
            Handles cases where input is a list of patches of images (N>1), or list of whole images (N=1)

            Inputs:
                -- 'x' (B x N x C x T x h x w), batch of images
            Outputs:
                -- 'feats' (B x C x T x N), node embeddings
                -- 'maps'  (B x N x C x T x H x W), node feature maps
        '''
        B, N, C, T, h, w = x.shape
        maps = self.encoder(x.flatten(0, 1))
        H, W = maps.shape[-2:]

        if self.featdrop_rate > 0:
            maps = self.featdrop(maps)

        if N == 1:  # flatten single image's feature map to get node feature 'maps'
            maps = maps.permute(0, -2, -1, 1, 2).contiguous()
            maps = maps.view(-1, *maps.shape[3:])[..., None, None]
            N, H, W = maps.shape[0] // B, 1, 1

        # compute node embeddings by spatially pooling node feature maps
        feats = maps.sum(-1).sum(-1) / (H*W)
        feats = self.selfsim_fc(feats.transpose(-1, -2)).transpose(-1,-2)
        feats = F.normalize(feats, p=2, dim=1)
    
        feats = feats.view(B, N, feats.shape[1], T).permute(0, 2, 3, 1)
        maps  =  maps.view(B, N, *maps.shape[1:])

        return feats, maps

    def forward(self, x, just_feats=False,):
        '''
        Input is B x T x N*C x H x W, where either
           N>1 -> list of patches of images
           N=1 -> list of images
        '''
        B, T, C, H, W = x.shape
        _N, C = C//3, 3
    
        #################################################################
        # Pixels to Nodes 
        #################################################################
        x = x.transpose(1, 2).view(B, _N, C, T, H, W)
        q, mm = self.pixels_to_nodes(x)
        B, C, T, N = q.shape

        if just_feats:
            h, w = np.ceil(np.array(x.shape[-2:]) / self.map_scale).astype(np.int)
            return (q, mm) if _N > 1 else (q, q.view(*q.shape[:-1], h, w))

        #################################################################
        # Compute walks 
        #################################################################
        walks = dict()
        As = self.affinity(q[:, :, :-1], q[:, :, 1:])
        A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)]

        #################################################### Palindromes
        if not self.sk_targets:  
            A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)]
            AAs = []
            for i in list(range(1, len(A12s))):
                g = A12s[:i+1] + A21s[:i+1][::-1]
                aar = aal = g[0]
                for _a in g[1:]:
                    aar, aal = aar @ _a, _a @ aal

                AAs.append((f"l{i}", aal) if self.flip else (f"r{i}", aar))
    
            for i, aa in AAs:
                walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]

        #################################################### Sinkhorn-Knopp Target (experimental)
        else:   
            a12, at = A12s[0], self.stoch_mat(A[:, 0], do_dropout=False, do_sinkhorn=True)
            for i in range(1, len(A12s)):
                a12 = a12 @ A12s[i]
                at = self.stoch_mat(As[:, i], do_dropout=False, do_sinkhorn=True) @ at
                with torch.no_grad():
                    targets = utils.sinkhorn_knopp(at, tol=0.001, max_iter=10, verbose=False).argmax(-1).flatten()
                walks[f"sk {i}"] = [a12, targets]

        #################################################################
        # Compute loss 
        #################################################################
        xents = [torch.tensor([0.]).to(self.args.device)]
        diags = dict()

        for name, (A, target) in walks.items():
            logits = torch.log(A+EPS).flatten(0, -2)
            loss = self.xent(logits, target).mean()
            acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            diags.update({f"{H} xent {name}": loss.detach(),
                          f"{H} acc {name}": acc})
            xents += [loss]

        #################################################################
        # Visualizations
        #################################################################
        if (np.random.random() < 0.02) and (self.vis is not None): # and False:
            with torch.no_grad():
                self.visualize_frame_pair(x, q, mm)
                if _N > 1: # and False:
                    self.visualize_patches(x, q)

        loss = sum(xents)/max(1, len(xents)-1)
        
        return q, loss, diags

    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]

    def visualize_patches(self, x, q):
        # all patches
        all_x = x.permute(0, 3, 1, 2, 4, 5)
        all_x = all_x.reshape(-1, *all_x.shape[-3:])
        all_f = q.permute(0, 2, 3, 1).reshape(-1, q.shape[1])
        all_f = all_f.reshape(-1, *all_f.shape[-1:])
        all_A = torch.einsum('ij,kj->ik', all_f, all_f)
        utils.visualize.nn_patches(self.vis.vis, all_x, all_A[None])

    def visualize_frame_pair(self, x, q, mm):
        t1, t2 = np.random.randint(0, q.shape[-2], (2))
        f1, f2 = q[:, :, t1], q[:, :, t2]

        A = self.affinity(f1, f2)
        A1, A2  = self.stoch_mat(A, False, False), self.stoch_mat(A.transpose(-1, -2), False, False)
        AA = A1 @ A2
        xent_loss = self.xent(torch.log(AA + EPS).flatten(0, -2), self.xent_targets(AA))

        utils.visualize.frame_pair(x, q, mm, t1, t2, A, AA, xent_loss, self.vis.vis)
