#coding:utf-8
"""
2018-12-16 10:41:42
pengfei
用于计算mAP
"""
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
class ComputemAP:

    def __init__(self, preds, labels, iou_thresh, class_list, use_cuda):
        """
        :param
            preds: size is (N, 7), 7 means [im_name, xmin_ ymin, xmax, ymax, class_name, confidence]
        :param
            labels: size is (N, 6), 6 means [im_name, xmin_ ymin, xmax, ymax, class_name]
        :param
            iou_thresh: IOU threshold to determine the TP (float)
        :param
            class_list: name list of class (tuple)
        :returns
           return an object
        """

        self.preds = preds
        self.labels = labels
        self.iou_thresh = iou_thresh
        self.class_list = class_list
        self.use_cuda = use_cuda

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
            mAP ， 最终结果
            all_ap , a list, 每个元素是一个dictionary with format  {class_name: ap_value}
        """

        mAP, all_ap = self.compute_map(self.preds,
                                       self.labels,
                                       self.iou_thresh,
                                       self.class_list)

        return mAP, all_ap

    def compute_map(self, preds, labels, iou_thresh, class_list):
        """
        :brief  计算mAP
        :param
            preds：[N, 5];  5: means [im_name, xmin, ymin, xmax, ymax, class_name, conf]
            labels:[N, 6];  6: means [im_name, xmin, ymin, xmax. ymax, class_name]
            class_list:  a tuple contain all class name
            iou_thresh: IOU thresh to determine TP
        :return:
            mAP ， 最终结果
            all_ap , a list, 每个元素是一个dictionary with format  {class_name: ap_value}

        """

        mAP = 0.0
        all_ap = []
        for class_name in class_list:
            ap = self.single_class_ap(preds, labels, class_name, iou_thresh)
            mAP += ap[class_name]
            all_ap.append(ap)  #  便于分类统计 AP值

        mAP = mAP / len(class_list)

        return mAP, all_ap

    def single_class_ap(self, preds, labels, class_name, iou_thresh):
        """
        :brief  计算单个类别的AP值
        :param
            preds：[N, 7];  7: means [im_name, xmin, ymin, xmax, ymax, class_name, conf]
            labels:[N, 6];  6: means [im_name, xmin, ymin, xmax. ymax, class_name]
            class_name:  name of calss
            iou_thresh: IOU thresh to determine TP
        :return:
            ap , a dictionary with format  {class_name: ap_value}

        :note
            about the data type
            xmin, ymin, xmax, ymax,  conf, iou_thresh ： float
            im_name, class_name : str
        """

        try:
            t1 = time.time()
            preds_spec_class = [x for x in preds if x[5] == class_name]
            labels_spec_class = [x for x in labels if x[5] == class_name]

        except:
            print(len(preds), preds[0])
            sys.exit(6)

        #  sort by confidence
        def base_confidence(elem):
            return elem[6]

        preds_spec_class.sort(key=base_confidence, reverse=True)



        preds_coords = [x[1:5] for x in preds_spec_class]
        gts_coords = [list(map(float, x[1:5])) for x in labels_spec_class]
        preds_coords = torch.tensor(preds_coords, dtype=torch.float)
        gts_coords = torch.tensor(gts_coords, dtype=torch.float)

        try:
            iou = self.compute_iou(preds_coords, gts_coords).cpu()
        except:
            idx = 0
            iou = torch.tensor([]).cpu()
            gap = 7000
            while True:
                if idx+gap <  preds_coords.size(0):
                    iou_temp = self.compute_iou(preds_coords[idx:idx+gap, :], gts_coords).cpu()
                    iou = torch.cat((iou, iou_temp), 0)
                    idx += gap
                    torch.cuda.empty_cache()
                else:
                    iou_temp = self.compute_iou(preds_coords[idx:, :], gts_coords).cpu()
                    iou = torch.cat((iou, iou_temp), 0)
                    torch.cuda.empty_cache()
                    break

        idx = torch.nonzero(iou.gt(0.5))
        for i in idx:
            if preds_spec_class[i[0]][0] == labels_spec_class[i[1]][0]:
                preds_spec_class[i[0]].append(1)


        """  compute the recall and precision """
        recall = np.zeros(len(preds_spec_class), dtype=np.float32)
        precision = np.zeros(len(preds_spec_class), dtype=np.float32);

        k = 0
        for idx, i_pred in enumerate(preds_spec_class):
            if len(i_pred)>=8 :
                k = k + 1
            recall[idx] = k / len(labels_spec_class)
            precision[idx] = k / (idx + 1)

        # plt.plot(recall, precision, color='r')
        # plt.show()

        """ compute the AP"""
        recall_spaced = np.arange(0, 1.1, 0.1)
        spaced_precision_sum = 0
        for i in range(11):

            idx = np.where(recall >= recall_spaced[i])
            if len(idx[0]) == 0:
                max_precision = 0
            else:
                max_precision = np.max(precision[idx[0][0]:])
            spaced_precision_sum += max_precision

        ap_val = spaced_precision_sum / 11

        ap = {class_name: ap_val}

        t2 = time.time()
        print(ap ,end='  ')
        print(t2-t1)
        return ap

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''

        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )


        wh = rb - lt  # [N,M,2]
        del rb
        del lt
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        del wh

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter.float() / (area1 + area2 - inter + 1e-5)

        return iou


if __name__ == "__main__":

    import glob
    import csv

    VOC_CLASSES = (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    # 原始 image name , confidience coordinate
    # size is (N, 7), 7 means [im_name, xmin_ ymin, xmax, ymax, class_name, confidence]
    result_path = "resultsVOC2007/results/"
    preds = []
    #  把detection的结果整理一下
    for class_name in VOC_CLASSES:

        with open(os.path.join(result_path, "det_test_" + class_name + ".txt")) as detect_file:

            lines = detect_file.readlines()
            for i in lines:
                temp = i.split(' ')
                temp[1:] = [float(x) for x in temp[1:]]
                temp.append(class_name) # add class_name
                temp.append(temp[1]) # append confidence
                temp.pop(1) # delete confidence
                preds.append(temp)

    # preds:[im_name, xmin, ymin, xmax, ymax, class_name, confidence]

    f = './CSV/voc/voc07_test.csv'
    with open(f) as f:
        labels = list(csv.reader(f))

    print(len(labels))

    ap = ComputemAP(preds=preds, labels=labels, class_list=VOC_CLASSES, iou_thresh=0.5)

    mAP, ap_list = ap()

    ap_val = []
    for i in ap_list:
        print(i)
        ap_val += list(i.values())
    print(mAP)



    ap_val = [round(x, 2) for x in ap_val]
    plt.figure()
    plt.barh(range(len(ap_val)), ap_val, height=0.7, color='steelblue', alpha=0.8)      # 从下往上画
    plt.yticks(range(len(ap_val)), list(VOC_CLASSES))
    plt.xlim(0, 1)
    plt.xlabel("AP")
    plt.title("AP for every class")
    print('nihao')
    for x, y in enumerate(ap_val):
        plt.text(y + 0.2, x - 0.1, '%s' % y)

    plt.show()
