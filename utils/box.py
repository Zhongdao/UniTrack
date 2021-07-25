###################################################################
# File Name: box.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Wed Dec 23 16:27:15 2020
###################################################################

import torch
import torchvision
import numpy as np


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = x.clone() if x.dtype is torch.float32 else x.copy()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone() if x.dtype is torch.float32 else x.copy()
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def tlwh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone() if x.dtype is torch.float32 else x.copy()
    y[:, 2] = (x[:, 0] + x[:, 2])
    y[:, 3] = (x[:, 1] + x[:, 3])
    return y


def tlwh_to_xywh(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret


def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= (ret[3] + 1e-6)
    return ret


def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret


def scale_box(scale, coords):
    c = coords.clone()
    c[:, [0, 2]] = coords[:, [0, 2]] * scale[0]
    c[:, [1, 3]] = coords[:, [1, 3]] * scale[1]
    return c


def scale_box_letterbox_size(img_size, coords, img0_shape):
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, 0:4] *= gain
    coords[:, [0, 2]] += pad_x
    coords[:, [1, 3]] += pad_y
    return coords


def scale_box_input_size(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    return coords


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes = np.asarray(boxes)
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.copy(boxes)
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def clip_box(bbox, im_shape):
    h, w = im_shape[:2]
    bbox = np.copy(bbox)
    bbox[0] = max(min(bbox[0], w - 1), 0)
    bbox[1] = max(min(bbox[1], h - 1), 0)
    bbox[2] = max(min(bbox[2], w - 1), 0)
    bbox[3] = max(min(bbox[3], h - 1), 0)

    return bbox


def int_box(box):
    box = np.asarray(box, dtype=np.float)
    box = np.round(box)
    return np.asarray(box, dtype=np.int)


def remove_duplicated_box(boxes, iou_th=0.5):
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    jac = torchvision.ops.box_iou(boxes, boxes).float()
    jac -= torch.eye(jac.shape[0])
    keep = np.ones(len(boxes)) == 1
    for i, b in enumerate(boxes):
        if b[0] == -1 and b[1] == -1 and b[2] == 10 and b[3] == 10:
            keep[i] = False
    for r, row in enumerate(jac):
        if keep[r]:
            discard = torch.where(row > iou_th)
            keep[discard] = False
    return np.where(keep)[0]


def skltn2box(skltn):
    dskltn = dict()
    for s in skltn:
        dskltn[s['id'][0]] = (int(s['x'][0]), int(s['y'][0]))
    if len(dskltn) == 0:
        return np.array(
                [-1, -1, np.random.randint(1, 40), np.random.randint(1, 70)])

    xmin = np.min([dskltn[k][0] for k in dskltn])
    xmax = np.max([dskltn[k][0] for k in dskltn])
    ymin = np.min([dskltn[k][1] for k in dskltn])
    ymax = np.max([dskltn[k][1] for k in dskltn])
    if xmin == xmax:
        xmax += 10
    if ymin == ymax:
        ymax += 10
    return np.array([xmin, ymin, xmax, ymax])
