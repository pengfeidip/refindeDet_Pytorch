"""
Time:2019-2-26
Author：pengfei
Motivation：for training on a csv dataset
"""
"""csv Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import csv
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd

class CSVAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, keep_difficult=False):

        self.keep_difficult = keep_difficult

    def __call__(self, targets, width, height, classes):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        wh = np.array([width, height, width, height, 1])
        for i in targets:
            bboxe = list(map(float, i[:-1]))  # str to float
            bboxe.append(classes[i[-1]])
            bboxe /= wh # bounding box normalization
            res += [list(bboxe)]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class CSVDataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): image path with all trainval and test images.
        csv_file (string): csv file which contain object's information
        class_file(string): csv file contains classes information
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'csv')
    """
    def __init__(self, csv_file,
                 classes_file,
                 transform=None, target_transform=CSVAnnotationTransform(),
                 dataset_name='csv'):
        self.csv_file = csv_file
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.img_lists = []
        self.targets = []

        # construct class information dict
        # key is class's name, item is class's index
        with open(classes_file) as f:
            temp = list(csv.reader(f))
        self.classes = dict(zip([x[0] for x in temp], [float(x[1]) for x in temp]))

        self.num_classes = len(temp)+1

        with open(csv_file) as f:
            temp = list(csv.reader(f))

        for i in temp:
            self.targets.append(i[1:])

        self.ids = self._get_im_lists(csv_file)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):

        im_info = self.ids[index]
        im_path = im_info[0]
        img = cv2.imread(im_path)
        # img = self.ims[index]
        height, width, channels = img.shape

        # Reshape the targets from [num_targets * 5] to [num_targets, 5]
        num_targets = int( (len(im_info)-1) / 5)
        targets = []
        for i in range(num_targets):
            targets.append(im_info[1+5*i:1+5*(i+1)])

        if self.target_transform is not None:
            target = self.target_transform(targets, width, height, self.classes)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_path = self.ids[index]
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        im_info = self.ids[index]

        # Reshape the targets from [num_targets * 5] to [num_targets, 5]
        num_targets = int((len(im_info) - 1) / 5)
        targets = []
        for i in range(num_targets):
            targets.append(im_info[1 + 5 * i:1 + 5 * (i + 1)])

        gt = self.target_transform(targets, 1, 1, self.classes)
        return targets, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


    def _get_im_lists(self, dataset_file):
        """
        :param dataset_file: (string)  csv file contains objects
        :return: (list) images list. Shape [num_images]
                [[im_name_1, x1,y1,x2,y2, classes_name, ....], ....]
        """
        im_list_old = pd.read_csv(dataset_file).values.tolist()

        im_list_new = []
        current_im_name = ''

        new_im = []
        for index, i in enumerate(im_list_old):
            if i[0] != current_im_name:
                if index > 0:
                    im_list_new.append(new_im)
                new_im = i
                current_im_name = i[0]
            else:
                new_im += i[1:]

            if index == len(im_list_old) - 1:
                im_list_new.append(new_im)

        return im_list_new