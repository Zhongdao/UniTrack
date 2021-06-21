# Video's features
import numpy as np
from sklearn.decomposition import PCA
import cv2
import imageio as io

import visdom
import time
import PIL
import torchvision
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
import pycocotools.mask as mask_utils


def pca_feats(ff, K=1, solver='auto', whiten=True, img_normalize=True):
    ## expect ff to be   N x C x H x W

    N, C, H, W = ff.shape
    pca = PCA(
        n_components=3*K,
        svd_solver=solver,
        whiten=whiten
    )

    ff = ff.transpose(1, 2).transpose(2, 3)
    ff = ff.reshape(N*H*W, C).numpy()
    
    pca_ff = torch.Tensor(pca.fit_transform(ff))
    pca_ff = pca_ff.view(N, H, W, 3*K)
    pca_ff = pca_ff.transpose(3, 2).transpose(2, 1)

    pca_ff = [pca_ff[:, kk:kk+3] for kk in range(0, pca_ff.shape[1], 3)]

    if img_normalize:
        pca_ff = [(x - x.min()) / (x.max() - x.min()) for x in pca_ff]

    return pca_ff[0] if K == 1 else pca_ff


def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        video = (video*255).astype(np.uint8)
        
    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration = 0.2)


def draw_matches(x1, x2, i1, i2):
    # x1, x2 = f1, f2/
    detach = lambda x: x.detach().cpu().numpy().transpose(1,2,0) * 255
    i1, i2 = detach(i1), detach(i2)
    i1, i2 = cv2.resize(i1, (400, 400)), cv2.resize(i2, (400, 400))

    for check in [True]:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=check)

        h = int(x1.shape[-1]**0.5)
        matches = bf.match(x1.t().cpu().detach().numpy(), x2.t().cpu().detach().numpy())

        scale = i1.shape[-2] / h
        grid = torch.stack([torch.arange(0, h)[None].repeat(h, 1), torch.arange(0, h)[:, None].repeat(1, h)])
        
        grid = grid.view(2, -1)
        grid = grid * scale + scale//2

        kps = [cv2.KeyPoint(grid[0][i], grid[1][i], 1) for i in range(grid.shape[-1])]

        matches = sorted(matches, key = lambda x:x.distance)

        # img1 = img2 = np.zeros((40, 40, 3))
        out = cv2.drawMatches(i1.astype(np.uint8), kps, i2.astype(np.uint8), kps,matches[:], None, flags=2).transpose(2,0,1)

    return out


import wandb
class Visualize(object):
    def __init__(self, args):

        self._env_name = args.name
        self.vis = visdom.Visdom(
            port=args.port,
            server='http://%s' % args.server,
            env=self._env_name,
        )
        self.args = args

        self._init = False

    def wandb_init(self, model):
        if not self._init:
            self._init = True
            wandb.init(project="videowalk", group="release", config=self.args)
            wandb.watch(model)

    def log(self, key_vals):
        return wandb.log(key_vals)

    def nn_patches(self, P, A_k, prefix='', N=10, K=20):
        nn_patches(self.vis, P, A_k, prefix, N, K)

    def save(self):
        self.vis.save([self._env_name])

def get_stride(im_sz, p_sz, res):
    stride = (im_sz - p_sz)//(res-1)
    return stride

def nn_patches(vis, P, A_k, prefix='', N=10, K=20):
    # produces nearest neighbor visualization of N patches given an affinity matrix with K channels

    P = P.cpu().detach().numpy()
    P -= P.min()
    P /= P.max()

    A_k = A_k.cpu().detach().numpy() #.transpose(-1,-2).numpy()
    # assert np.allclose(A_k.sum(-1), 1)

    A = np.sort(A_k, axis=-1)
    I = np.argsort(-A_k, axis=-1)
    
    vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header' %(prefix))

    for n,i in enumerate(np.random.permutation(P.shape[0])[:N]):
        p = P[i]
        vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header_%s' % (prefix, n))
        # vis.image(p,  win='%s_patch_query_%s' % (prefix, n))

        for k in range(I.shape[0]):
            vis.images(P[I[k, i, :K]], nrow=min(I.shape[-1], 20), win='%s_patch_values_%s_%s' % (prefix, n, k))
            vis.bar(A[k, i][::-1][:K], opts=dict(height=150, width=500), win='%s_patch_affinity_%s_%s' % (prefix, n, k))


def compute_flow(corr):
    # assume batched affinity, shape N x H * W x W x H
    h = w = int(corr.shape[-1] ** 0.5)

    # x1 -> x2
    corr = corr.transpose(-1, -2).view(*corr.shape[:-1], h, w)
    nnf = corr.argmax(dim=1)

    u = nnf % w # nnf.shape[-1]
    v = nnf / h # nnf.shape[-2] # nnf is an IntTensor so rounds automatically

    rr = torch.arange(u.shape[-1])[None].long().cuda()

    for i in range(u.shape[-1]):
        u[:, i] -= rr

    for i in range(v.shape[-1]):
        v[:, :, i] -= rr

    return u, v

def vis_flow_plt(u, v, x1, x2, A):
    flows = torch.stack([u, v], dim=-1).cpu().numpy()
    I, flows = x1.cpu().numpy(), flows[0]

    H, W = flows.shape[:2]
    Ih, Iw, = I.shape[-2:]
    mx, my = np.mgrid[0:Ih:Ih/(H+1), 0:Iw:Iw/(W+1)][:, 1:, 1:]
    skip = (slice(None, None, 1), slice(None, None, 1))

    ii = 0
    fig, ax = plt.subplots()
    im = ax.imshow((I.transpose(1,2,0)),)
    
    C = cm.jet(torch.nn.functional.softmax((A * A.log()).sum(-1).cpu(), dim=-1))
    ax.quiver(my[skip], mx[skip], flows[...,0][skip], flows[...,1][skip]*-1, C)#, scale=1, scale_units='dots')
    # ax.quiver(mx[skip], my[skip], flows[...,0][skip], flows[...,1][skip])

    return plt
    
def frame_pair(x, ff, mm, t1, t2, A, AA, xent_loss, viz):
    normalize = lambda xx: (xx-xx.min()) / (xx-xx.min()).max()
    spatialize = lambda xx: xx.view(*xx.shape[:-1], int(xx.shape[-1]**0.5), int(xx.shape[-1]**0.5))

    N = AA.shape[-1]
    H = W = int(N**0.5)
    AA = AA.view(-1, H * W, H, W)

    ##############################################
    ## Visualize PCA of Embeddings, Correspondences
    ##############################################

    # import pdb; pdb.set_trace()
    if (len(x.shape) == 6 and x.shape[1] == 1):
        x = x.squeeze(1)

    if len(x.shape) < 6:   # Single image input, no patches
        # X here is B x C x T x H x W
        x1, x2 = normalize(x[0, :, t1]), normalize(x[0, :, t2])
        f1, f2 = ff[0, :, t1], ff[0, :, t2]
        ff1 , ff2 = spatialize(f1), spatialize(f2)

        xx = torch.stack([x1, x2]).detach().cpu()
        viz.images(xx, win='imgs')

        # Flow
        u, v = compute_flow(A[0:1])
        flow_plt = vis_flow_plt(u, v, x1, x2, A[0])
        viz.matplot(flow_plt, win='flow_quiver')

        # Keypoint Correspondences
        kp_corr = draw_matches(f1, f2, x1, x2)
        viz.image(kp_corr, win='kpcorr')

        # # PCA VIZ
        pca_ff = pca_feats(torch.stack([ff1,ff2]).detach().cpu())
        pca_ff = make_gif(pca_ff, outname=None)
        viz.images(pca_ff.transpose(0, -1, 1, 2), win='pcafeats', opts=dict(title=f"{t1} {t2}"))

    else:  # Patches as input
        # X here is B x N x C x T x H x W
        x1, x2 =  x[0, :, :, t1],  x[0, :, :, t2]
        m1, m2 = mm[0, :, :, t1], mm[0, :, :, t2]

        pca_ff = pca_feats(torch.cat([m1, m2]).detach().cpu())
        pca_ff = make_gif(pca_ff, outname=None, sz=64).transpose(0, -1, 1, 2)
        
        pca1 = torchvision.utils.make_grid(torch.Tensor(pca_ff[:N]), nrow=int(N**0.5), padding=1, pad_value=1)
        pca2 = torchvision.utils.make_grid(torch.Tensor(pca_ff[N:]), nrow=int(N**0.5), padding=1, pad_value=1)
        img1 = torchvision.utils.make_grid(normalize(x1)*255, nrow=int(N**0.5), padding=1, pad_value=1)
        img2 = torchvision.utils.make_grid(normalize(x2)*255, nrow=int(N**0.5), padding=1, pad_value=1)
        viz.images(torch.stack([pca1,pca2]), nrow=4, win='pca_viz_combined1')
        viz.images(torch.stack([img1.cpu(),img2.cpu()]), opts=dict(title=f"{t1} {t2}"), nrow=4, win='pca_viz_combined2')
    
    ##############################################
    # LOSS VIS
    ##############################################
    color = cm.get_cmap('winter')

    xx = normalize(xent_loss[:H*W])
    img_grid = [cv2.resize(aa, (50,50), interpolation=cv2.INTER_NEAREST)[None] 
                for aa in AA[0, :, :, :, None].cpu().detach().numpy()]
    img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H*W)]
    img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H*W)]
    img_grid = torch.from_numpy(np.array(img_grid))
    img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)
    
    viz.images(img_grid, win='lossvis')

def get_color(idx):
    idx = idx * 17
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, obs, obj_ids, scores=None, frame_id=0, fps=0.):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 150.))
    alpha = 0.4

    for i, ob in enumerate(obs): 
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(obj_id)
        if len(ob) == 4:
            x1, y1, w, h = ob
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
        elif isinstance(ob, dict):
            mask = mask_utils.decode(ob)
            mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)[:,:,None]
            mask_color = mask * color
            im = (1 - mask) * im + mask * (alpha*im + (1-alpha)*mask_color) 
        else:
            raise ValueError('Observation format not supported.')
    return im


def vis_pose(oriImg, points):

    pa = np.zeros(15)
    pa[2] = 0
    pa[12] = 8
    pa[8] = 4
    pa[4] = 0
    pa[11] = 7
    pa[7] = 3
    pa[3] = 0
    pa[0] = 1
    pa[14] = 10
    pa[10] = 6
    pa[6] = 1
    pa[13] = 9
    pa[9] = 5
    pa[5] = 1

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[0, :]
    y = points[1, :]

    for n in range(len(x)):
        pair_id = int(pa[n])

        x1 = int(x[pair_id])
        y1 = int(y[pair_id])
        x2 = int(x[n])
        y2 = int(y[n])

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            cv2.line(canvas, (x1, y1), (x2, y2), colors[n], 8)

    return canvas


def draw_skeleton(aa, kp, color, show_skeleton_labels=False, dataset= "PoseTrack"):
    if dataset == "COCO":
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], 
                [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',
                    'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                    'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    elif dataset == "PoseTrack":
        skeleton = [[10, 11], [11, 12], [9,8], [8,7],
                    [10, 13], [9, 13], [13, 15], [10,4],
                    [4,5], [5,6], [9,3], [3,2], [2,1]]
        kp_names = ['right_ankle', 'right_knee', 'right_pelvis',
                    'left_pelvis', 'left_knee', 'left_ankle',
                    'right_wrist', 'right_elbow', 'right_shoulder',
                    'left_shoulder', 'left_elbow', 'left_wrist',
                    'upper_neck', 'nose', 'head']
    for i, j in skeleton:
        if kp[i-1][0] >= 0 and kp[i-1][1] >= 0 and kp[j-1][0] >= 0 and kp[j-1][1] >= 0 and \
            (len(kp[i-1]) <= 2 or (len(kp[i-1]) > 2 and  kp[i-1][2] > 0.1 and kp[j-1][2] > 0.1)):
            st = (int(kp[i-1][0]), int(kp[i-1][1]))
            ed = (int(kp[j-1][0]), int(kp[j-1][1]))
            cv2.line(aa, st, ed,  color, max(1, int(aa.shape[1]/150.)))
    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:
            pt = (int(kp[j][0]), int(kp[j][1]))
            if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                cv2.circle(aa, pt, 2, tuple((0,0,255)), 2)
            elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                cv2.circle(aa, pt, 2, tuple((255,0,0)), 2)

            if show_skeleton_labels and (len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1)):
                cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
