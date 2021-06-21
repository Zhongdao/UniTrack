import os
import argparse
import numpy as np
from PIL import Image

n_cl = 20
CLASSES = ['background', 'hat', 'hair', 'sun-glasses', 'upper-clothes', 'dress',
           'coat', 'socks', 'pants', 'gloves', 'scarf', 'skirt', 'torso-skin',
           'face', 'right-arm', 'left-arm', 'right-leg', 'left-leg', 'right-shoe', 'left-shoe']

GT_DIR = '/scratch/ajabri/data/VIP/DATA/Category_ids/'
PRE_DIR = '/tmp/vip/'

parser = argparse.ArgumentParser()

parser.add_argument("-g", "--gt_dir", type=str, default=GT_DIR,
                    help="ground truth path")
parser.add_argument("-p", "--pre_dir", type=str, default=PRE_DIR,
                    help="prediction path")

args = parser.parse_args()

def main():
    image_paths, label_paths = init_path()
    hist = compute_hist(image_paths, label_paths)
    show_result(hist)

def _get_voc_color_map(n=256):
    color_map = np.zeros((n, 3))
    index_map = {}
    for i in range(n):
        r = b = g = 0
        cid = i
        for j in range(0, 8):
            r = np.bitwise_or(r, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-1], 7-j))
            g = np.bitwise_or(g, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-2], 7-j))
            b = np.bitwise_or(b, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-3], 7-j))
            cid = np.right_shift(cid, 3)

        color_map[i][0] = r
        color_map[i][1] = g
        color_map[i][2] = b
        index_map['%d_%d_%d'%(r, g, b)] = i
    return color_map, index_map

def init_path():
    image_dir = args.pre_dir
    label_dir = args.gt_dir

    file_names = []
    for vid in os.listdir(image_dir):
        for img in os.listdir(os.path.join(image_dir, vid)):
            file_names.append([vid, img])
    print ("result of", image_dir)

    image_paths = []
    label_paths = []
    for file_name in file_names:
        image_paths.append(os.path.join(image_dir, file_name[0], file_name[1]))
        label_paths.append(os.path.join(label_dir, file_name[0], file_name[1]))
    return image_paths, label_paths

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

import sys

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
import cv2

def compute_hist(images, labels):

    color_map, index_map = _get_voc_color_map()
    hist = np.zeros((n_cl, n_cl))
    for img_path, label_path in zip(images, labels):
        label = Image.open(label_path.replace('.jpg', '.png'))
        label_array = np.array(label, dtype=np.int32)
        image = cv2.imread(img_path)
        image = Image.fromarray(image)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape

        imgsz = image_array.shape

        if image_array.max() > 20:
            print(img_path, image_array.max())
            # continue
            import pdb; pdb.set_trace()


        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.NEAREST)
            image_array = np.array(image, dtype=np.int32)

        if len(image_array.shape) == 3:
            image_array = image_array[..., -1]

        # import pdb; pdb.set_trace()

        hist += fast_hist(label_array, image_array, n_cl)

    return hist

def show_result(hist):

    classes = CLASSES
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print ('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print ('>>>', 'overall accuracy', acc)
    print ('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    print ('Accuracy for each class (pixel accuracy):')
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print ('>>>', 'mean accuracy', np.nanmean(acc))
    print ('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print ('>>>', 'mean IU', np.nanmean(iu))
    print ('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print ('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print ('=' * 50)



if __name__ == '__main__':
    main()