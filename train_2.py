"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from data import CSVDataset, CSVAnnotationTransform
import torch.utils.data as data
from tqdm import  tqdm
import pandas as pd

from models.refinedet import build_refinedet
from mAP import ComputemAP

import sys
import os
import time
import argparse
import numpy as np
import pickle
import csv
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
# './backup_weights/RefineDet320_CSV_original.pth'
#./weights/RefineDet320_CSV_final.pth
parser.add_argument('--trained_model',
                    default='./focal_loss_weights/RefineDet320_CSV_55000.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default='./results',
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--input_size', default='320', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


dataset_mean = (104, 117, 123)


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(net, dataset, gt_file):

    predictions = pd.read_csv('pred.csv', header=None).values.tolist()

    # save predictions in current path


    print('Evaluating detections')
    evaluate_detections(predictions, dataset, gt_file)

def evaluate_detections(preds, dataset, gt_file):

    classes = list(dataset.classes.keys())

    with open(gt_file) as f:
        labels = list(csv.reader(f))

    print("Computing the mAp ing...... please wait a moment")
    ap = ComputemAP(preds=preds, labels=labels, class_list=classes, iou_thresh=0.5, use_cuda=False)

    mAP, ap_list = ap()
    for i in ap_list:
        print(i)
    print(mAP)




if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    gt_file = "./csv/voc/0712_trainval.csv"

    dataset = CSVDataset(csv_file=gt_file,
                         classes_file='./csv/voc/classes.csv',
                         transform=BaseTransform(int(args.input_size),
                                                 dataset_mean))

    net = build_refinedet('test', int(args.input_size), dataset.num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(net,  dataset, gt_file)
