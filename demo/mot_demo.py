###################################################################
# File Name: mot_demo.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Sat Jul 24 16:07:23 2021
###################################################################

import os
import sys
import yaml
import argparse
import os.path as osp
from loguru import logger

import cv2
import torch
import numpy as np
from torchvision.transforms import transforms as T

sys.path[0] = os.getcwd()
from data.video import LoadVideo
from utils.meter import Timer
from utils import visualize as vis
from detector.YOLOX.yolox.exp import get_exp
from detector.YOLOX.yolox.utils import get_model_info
from detector.YOLOX.yolox.data.datasets import COCO_CLASSES
from detector.YOLOX.tools.demo import Predictor

from utils.box import scale_box_input_size
from tracker.mot.box import BoxAssociationTracker


def make_parser():
    parser = argparse.ArgumentParser("YOLOX + UniTrack MOT demo")
    # Common arguments
    parser.add_argument('--demo', default='video',
                        help='demo type, eg. video or webcam')
    parser.add_argument('--path', default='./docs/test_video.mp3',
                        help='path to images or video')
    parser.add_argument('--camid', type=int, default=0,
                        help='webcam demo camera id')
    parser.add_argument('--save_result', action='store_true',
                        help='whether to save result')
    parser.add_argument("--nms", default=None, type=float,
                        help="test nms threshold")
    parser.add_argument("--tsize", default=[640, 480], type=int, nargs='+',
                        help="test img size")
    parser.add_argument("--exp_file", type=str,
                        default='./detector/YOLOX/exps/default/yolox_x.py',
                        help="pls input your expriment description file")
    parser.add_argument('--output-root', default='./results/mot_demo',
                        help='output directory')
    parser.add_argument('--classes', type=int, nargs='+',
                        default=list(range(90)), help='COCO_CLASSES')

    # Detector related
    parser.add_argument("-c", "--ckpt",  type=str,
                        default='./detector/YOLOX/weights/yolox_x.pth',
                        help="model weights of the detector")
    parser.add_argument("--conf", default=0.65, type=float,
                        help="detection confidence threshold")

    # UniTrack related
    parser.add_argument('--config', type=str, help='tracker config file',
                        default='./config/imagenet_resnet18_s3.yaml')

    return parser


def dets2obs(dets, imginfo, cls):
    if dets is None or len(dets) == 0:
        return np.array([])
    obs = dets.cpu().numpy()
    h, w = imginfo['height'], imginfo['width']
    # To xywh
    ret = np.zeros((len(obs), 6))
    ret[:, 0] = (obs[:, 0] + obs[:, 2]) * 0.5 / w
    ret[:, 1] = (obs[:, 1] + obs[:, 3]) * 0.5 / h
    ret[:, 2] = (obs[:, 2] - obs[:, 0]) / w
    ret[:, 3] = (obs[:, 3] - obs[:, 1]) / h
    ret[:, 4] = obs[:, 4] * obs[:, 5]
    ret[:, 5] = obs[:, 6]

    ret = [r for r in ret if int(r[5]) in cls]
    ret = np.array(ret)

    return ret


def eval_seq(opt, dataloader, detector, tracker,
             result_filename, save_dir=None,
             show_image=True):
    transforms = T.Compose([T.ToTensor(),
                            T.Normalize(opt.im_mean, opt.im_std)])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    timer = Timer()
    results = []
    for frame_id, (_, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1./max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        det_outputs, img_info = detector.inference(img)
        img = img / 255.
        img = transforms(img)
        obs = dets2obs(det_outputs[0], img_info, opt.classes)
        if len(obs) == 0:
            online_targets = []
        else:
            online_targets = tracker.update(img, img0, obs)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(
                    img0, online_tlwhs, online_ids, frame_id=frame_id,
                    fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(
                save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    return frame_id, timer.average_time, timer.calls


def main(exp, args):
    logger.info("Args: {}".format(args))

    # Data, I/O
    dataloader = LoadVideo(args.path, args.tsize)
    video_name = osp.basename(args.path).split('.')[0]
    result_root = osp.join(args.output_root, video_name)
    result_filename = os.path.join(result_root, 'results.txt')
    args.frame_rate = dataloader.frame_rate

    # Detector init
    det_model = exp.get_model()
    logger.info("Model Summary: {}".format(
        get_model_info(det_model, exp.test_size)))
    det_model.cuda()
    det_model.eval()
    logger.info("loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # load the model state dict
    det_model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    detector = Predictor(det_model, exp, COCO_CLASSES, None, None, 'gpu')

    # Tracker init
    tracker = BoxAssociationTracker(args)

    frame_dir = osp.join(result_root, 'frame')
    try:
        eval_seq(args, dataloader, detector, tracker, result_filename,
                 save_dir=frame_dir, show_image=False)
    except Exception as e:
        print(e)

    output_video_path = osp.join(result_root, video_name+'.avi')
    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
            osp.join(result_root, 'frame'), output_video_path)
    os.system(cmd_str)


if __name__ == '__main__':
    args = make_parser().parse_args()
    with open(args.config) as f:
        common_args = yaml.load(f)
    for k, v in common_args['common'].items():
        setattr(args, k, v)
    for k, v in common_args['mot'].items():
        setattr(args, k, v)
    exp = get_exp(args.exp_file, None)
    if args.conf is not None:
        args.conf_thres = args.conf
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = args.tsize[::-1]
        args.img_size = args.tsize
    args.classes = [x for x in args.classes]
    main(exp, args)
