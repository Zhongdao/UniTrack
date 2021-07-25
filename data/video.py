import os
import pdb
import glob
import json
import os.path as osp

import cv2
import numpy as np

import pycocotools.mask as mask_utils
from utils.box import xyxy2xywh

from torchvision.transforms import transforms as T


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower()
                                     in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path) 
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = self.get_size(self.vw, self.vh, self.width, self.height)
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh*a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img, dtype=np.float32)

        return self.count, img, img0
   
    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndObs: 
    def __init__(self, path, opt):
        obid = opt.obid
        img_size = getattr(opt,'img_size', None)
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.img_files = sorted(glob.glob('%s/*.*' % path))
            self.img_files = list(filter(
                lambda x: os.path.splitext(x)[1].lower() in image_format, self.img_files))
        elif os.path.isfile(path):
            self.img_files = [path,]

        self.label_files = [x.replace('images', osp.join('obs', obid)).replace(
            '.png', '.txt').replace('.jpg', '.txt') for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(opt.im_mean, opt.im_std)])
        self.use_lab = getattr(opt, 'use_lab', False)
        if not img_size is None:
            self.width = img_size[0]
            self.height = img_size[1]

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img_ori = cv2.imread(img_path)  # BGR
        if img_ori is None:
            raise ValueError('File corrupt {}'.format(img_path))

        h, w, _ = img_ori.shape
        img, ratio, padw, padh = letterbox(img_ori, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 0] = ratio * w * (labels0[:, 0] - labels0[:, 2] / 2) + padw
            labels[:, 1] = ratio * h * (labels0[:, 1] - labels0[:, 3] / 2) + padh
            labels[:, 2] = ratio * w * (labels0[:, 0] + labels0[:, 2] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 1] + labels0[:, 3] / 2) + padh
        else:
            labels = np.array([])
        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 0:4] = xyxy2xywh(labels[:, 0:4].copy())
            labels[:, 0] /= width
            labels[:, 1] /= height
            labels[:, 2] /= width
            labels[:, 3] /= height
       
        if self.use_lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = np.array([img[:, :, 0], ]*3)
            img = img.transpose(1, 2, 0)
        img = img / 255.
        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_ori, (h, w)

    def __len__(self):
        return self.nF  # number of batches

class LoadImagesAndObsTAO:
    def __init__(self, root, video_meta, obs, opt):
        self.dataroot = root
        self.img_ind = [x['id'] for x in video_meta]
        self.img_files = [x['file_name'] for x in video_meta]
        self.img_files = [osp.join(root, 'frames', x) for x in self.img_files]
        self.obs = [obs.get(x, []) for x in self.img_ind]
        self.use_lab = getattr(opt, 'use_lab', False)
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(opt.im_mean, opt.im_std)])

    def __getitem__(self, index):
        img_ori = cv2.imread(self.img_files[index])
        if img_ori is None:
            raise ValueError('File corrupt {}'.format(img_path))

        h, w, _ = img_ori.shape
        img = img_ori
        if self.use_lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = np.array([img[:,:,0],]*3)
            img = img.transpose(1,2,0)
        img = img / 255.
        img = np.ascontiguousarray(img[ :, :, ::-1]) # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)

        obs = self.obs[index]
        if len(obs) == 0:
            labels = np.array([[0,0,1,1,-1,-1]])
        else:
            boxes = np.array([x.get('bbox', [0,0,1,1]) for x in obs])
            scores = np.array([x.get('score', 0) for x in obs])[:, None]
            cat_ids = np.array([x.get('category_id',-1) for x in obs])[:, None]
            labels = np.concatenate([boxes, scores, cat_ids], axis=1)
            if len(labels) > 0:
                # From tlwh to xywh: (x,y) is the box center
                labels[:, 0] = labels[:, 0] + labels[:, 2] / 2
                labels[:, 1] = labels[:, 1] + labels[:, 3] / 2
                labels[:, 0] /= w
                labels[:, 1] /= h
                labels[:, 2] /= w
                labels[:, 3] /= h

        return img, labels, img_ori, (h,w)

    def __len__(self):
        return len(self.img_files)



class LoadImagesAndMaskObsVIS:
    def __init__(self, path, info, obs, opt):
        self.dataroot = path
        self.nF = info['length']
        self.img_files = [osp.join(path, p) for p in info['file_names']]
        self.obsbyobj = obs
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(opt.im_mean, opt.im_std)])
        self.use_lab = getattr(opt, 'use_lab', False)


    def __getitem__(self, idx):
        img_ori = cv2.imread(self.img_files[idx])
        if img_ori is None:
            raise ValueError('File corrupt {}'.format(img_path))

        h, w, _ = img_ori.shape
        img = img_ori
        if self.use_lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = np.array([img[:,:,0],]*3)
            img = img.transpose(1,2,0)
        img = img / 255.
        img = np.ascontiguousarray(img[ :, :, ::-1]) # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)


        labels = list()
        for obj in self.obsbyobj:
            RLE = obj['segmentations'][idx]
            if RLE: labels.append(mask_utils.decode(RLE))
            else: labels.append(np.zeros((h, w), dtype=np.uint8))
        labels = np.stack(labels)

        return img, labels, img_ori, (h, w)

    def __len__(self):
        return self.nF

    
class LoadImagesAndMaskObsMOTS(LoadImagesAndObs): 
    def __init__(self, path, opt):
        super(LoadImagesAndMaskObsMOTS, self).__init__(path, opt)

    def get_data(self, img_path, label_path):
        img_ori = cv2.imread(img_path)  # BGR
        if img_ori is None:
            raise ValueError('File corrupt {}'.format(img_path))
        h, w, _ = img_ori.shape

        img = img_ori
        if self.use_lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = np.array([img[:,:,0],]*3)
            img = img.transpose(1,2,0)
        img = img / 255.
        img = np.ascontiguousarray(img[ :, :, ::-1]) # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    labels.append(line.strip().split())
        nL = len(labels)
        if nL > 0:
            labels = [{'size':(int(h),int(w)), 'counts':m} for \
                    _, _,cid,h,w,m in labels if cid=='2']
            labels = [mask_utils.decode(rle) for rle in labels]
        labels = np.stack(labels) 
        return img, labels, img_ori, (h, w)


class LoadImagesAndPoseObs(LoadImagesAndObs): 
    def __init__(self, obs_jpath, opt):
        fjson = open(obs_jpath, 'r')
        self.infoj = json.load(fjson)['annolist']
        self.dataroot = opt.data_root
        self.nF = len(self.infoj)
        self.img_files = [osp.join(opt.data_root, p['image'][0]['name']) for p in self.infoj]
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(opt.im_mean, opt.im_std)])
        self.use_lab = getattr(opt, 'use_lab', False)

    def __getitem__(self, idx):
        img_ori = cv2.imread(self.img_files[idx])
        if img_ori is None:
            raise ValueError('File corrupt {}'.format(img_path))

        h, w, _ = img_ori.shape
        img = img_ori
        
        if self.use_lab:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = np.array([img[:,:,0],]*3)
            img = img.transpose(1,2,0)
        
        img = img / 255.
        img = np.ascontiguousarray(img[ :, :, ::-1]) # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)

        info_label = self.infoj[idx]['annorect']
        nobj = len(info_label)
        labels = list() 
        labels = [l['annopoints'][0]['point'] for l in info_label]

        return img, labels, img_ori, (h, w)
        


def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular 
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

