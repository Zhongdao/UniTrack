from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
import numpy as np
import os
import cv2

import scipy.io as sio

filelist = '/home/wangzd/datasets/GOT/JHMDB/split.txt'
src_folder = 'results/poseprop/womotion_resnet18_s3/'

f = open(filelist, 'r')
gts = []
heights = []
widths  = []
preds = []
jnt_visible_set = []
human_boxes = []

feat_res = 40

for cnt, line in enumerate(f):
    rows = line.strip().split()
    lblpath = rows[0] #+ '/joint_positions.mat'
    lbls_mat = sio.loadmat(lblpath)
    lbls_coord = lbls_mat['pos_img']
    lbls_coord = lbls_coord - 1

    gts.append(lbls_coord)

    imgpath = rows[1] + '/00001.png'
    img = cv2.imread(imgpath)
    heights.append(img.shape[0])
    widths.append(img.shape[1])

f.close()

# gts = gts[0: 200]
print('read gt')

# read prediction results
for i in range(len(gts)):

    # import pdb; pdb.set_trace()
    predfile = src_folder + str(i) + '.dat'
    predres  = np.load(predfile, allow_pickle=True)

    # import pdb; pdb.set_trace()

    jnt_visible = np.ones((predres.shape[1], predres.shape[2]))

    for j in range(predres.shape[1]):
        for k in range(predres.shape[2]):
            if predres[0, j, k] < 0:
                jnt_visible[j, k] = 0

    jnt_visible_set.append(jnt_visible)

    now_height = heights[i]
    now_width  = widths[i]
    predres[0, :, :] = predres[0, :, :] / float(feat_res) * now_width
    predres[1, :, :] = predres[1, :, :] / float(feat_res) * now_height

    preds.append(predres)

print('read prediction')

# compute the human box for normalization
for i in range(len(gts)):

    nowgt = gts[i]
    jnt_visible = jnt_visible_set[i]
    now_boxes = np.zeros(nowgt.shape[2])

    for k in range(nowgt.shape[2]):
        minx = 1e6
        maxx = -1
        miny = 1e6
        maxy = -1
        for j in range(nowgt.shape[1]):

            if jnt_visible[j, k] == 0:
                continue

            minx = np.min([minx, nowgt[0, j, k]])
            miny = np.min([miny, nowgt[1, j, k]])
            maxx = np.max([maxx, nowgt[0, j, k]])
            maxy = np.max([maxy, nowgt[1, j, k]])

        now_boxes[k] = 0.6 * np.linalg.norm(np.subtract([maxx,maxy],[minx,miny]))
        # now_boxes[k] = np.max([maxy - miny, maxx - minx])


    human_boxes.append(now_boxes)

print('done box')

# compute distances
distAll = {}
for pidx in range(15):
    distAll[pidx] = np.zeros([0,0])

for i in range(len(gts)):

    predres = preds[i]
    nowgt = gts[i]
    now_boxes = human_boxes[i]
    jnt_visible = jnt_visible_set[i]
    for j in range(nowgt.shape[1]):
        for k in range(nowgt.shape[2]):

            if jnt_visible[j, k] == 0:
                continue

            if k == 0:
                continue

            predx = predres[0, j, k]
            predy = predres[1, j, k]
            gtx   = nowgt[0, j, k]
            gty   = nowgt[1, j, k]
            d = np.linalg.norm(np.subtract([predx, predy],[gtx, gty]))
            dNorm = d / now_boxes[k]

            distAll[j] = np.append(distAll[j],[[dNorm]])

print('done distances')

def computePCK(distAll,distThresh):

    pckAll = np.zeros([len(distAll)+1,1])
    nCorrect = 0
    nTotal = 0
    for pidx in range(len(distAll)):
        idxs = np.argwhere(distAll[pidx] <= distThresh)
        pck = 100.0*len(idxs)/len(distAll[pidx])
        pckAll[pidx,0] = pck
        nCorrect += len(idxs)
        nTotal   += len(distAll[pidx])

    pckAll[len(distAll),0] = np.mean(pckAll[0 :len(distAll),0]) # 100.0*nCorrect/nTotal

    return pckAll

rng = [0.1, 0.2, 0.3, 0.4, 0.5]

for i in range(len(rng)):
    pckall = computePCK(distAll, rng[i])
    print(str(rng[i]) + ': ' + str(pckall[-1]) )
    # print(pckall[-1])
