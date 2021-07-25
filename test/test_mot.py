import os
import sys
import cv2
import yaml
import logging
import argparse
import os.path as osp

import numpy as np

sys.path[0] = os.getcwd()
from utils.log import logger
from utils.meter import Timer
import data.video as videodataset
from utils import visualize as vis
from utils.io import write_mot_results, mkdir_if_missing
from eval import trackeval

from tracker.mot.box import BoxAssociationTracker


def eval_seq(opt, dataloader, result_filename, save_dir=None,
             show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    opt.frame_rate = frame_rate
    tracker = BoxAssociationTracker(opt)
    timer = Timer()
    results = []
    for frame_id, (img, obs, img0, _) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1./max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        online_targets = tracker.update(img, img0, obs)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
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
    # save results
    write_mot_results(result_filename, results)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join('results/mot', exp_name, 'quantitive')
    mkdir_if_missing(result_root)

    # run tracking
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join('results/mot', exp_name, 'qualitative', seq)\
                    if save_images or save_videos else None

        logger.info('start seq: {}'.format(seq))
        dataloader = videodataset.LoadImagesAndObs(
                osp.join(data_root, seq, 'img1'), opt)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate')+10:
                                   meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, result_filename,
                              save_dir=output_dir, show_image=show_image,
                              frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
                    output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
        all_time, 1.0 / avg_time))

    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}

    eval_config['LOG_ON_ERROR'] = osp.join(result_root, 'error.log')
    eval_config['PLOT_CURVES'] = False
    dataset_config['GT_FOLDER'] = data_root
    dataset_config['SEQMAP_FOLDER'] = osp.join(data_root, '../../seqmaps')
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['TRACKERS_FOLDER'] = osp.join(result_root, '..')
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['TRACKERS_TO_EVAL'] = ['quantitive']
    dataset_config['BENCHMARK'] = 'MOT16'

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                   trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', required=True, type=str)
    parser.add_argument('--obid', default='FairMOT', type=str)
    opt = parser.parse_args()
    with open(opt.config) as f:
        common_args = yaml.load(f)
    for k, v in common_args['common'].items():
        setattr(opt, k, v)
    for k, v in common_args['mot'].items():
        setattr(opt, k, v)
    print(opt, end='\n\n')

    if not opt.test_mot16:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13
                    '''
        data_root = '{}/images/train'.format(opt.mot_root)
    else:
        seqs_str = '''MOT16-01
                     MOT16-03
                     MOT16-06
                     MOT16-07
                     MOT16-08
                     MOT16-12
                     MOT16-14'''
        seqs_str = '''MOT16-03'''
        data_root = '{}/images/test'.format(opt.mot_root)
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.exp_name,
         show_image=False,
         save_images=opt.save_images,
         save_videos=opt.save_videos)
