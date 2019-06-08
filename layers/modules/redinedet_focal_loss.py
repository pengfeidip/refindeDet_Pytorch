# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, refine_match


class RefineDetFocalLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, theta=0.01, use_ARM=False):
        super(RefineDetFocalLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.theta = theta # used in ARM for selecting boxes contains object
        self.use_ARM = use_ARM

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors = predictions
        alpha = 0.25
        gamma = 2
        #print(arm_loc_data.size(), arm_conf_data.size(),
        #      odm_loc_data.size(), odm_conf_data.size(), priors.size())
        #input()
        if self.use_ARM:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        #print(loc_data.size(), conf_data.size(), priors.size())

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if num_classes == 2:
                labels = labels >= 0
            defaults = priors.data
            if self.use_ARM:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx, arm_loc_data[idx].data)
            else:
                refine_match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        #loc_t = Variable(loc_t, requires_grad=False)
        #conf_t = Variable(conf_t, requires_grad=False)
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        #print(loc_t.size(), conf_t.size())

        if self.use_ARM:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_tmp = P[:,:,1]
            object_score_index = arm_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0
        #print(pos.size())
        #num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # print(loss_c.size())

        # Hard Negative Mining for location loss
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)

        # Focal loss for classification
        loss_c = torch.tensor([0.])
        # temp_loss_c = torch.tensor([0.])
        for i in range(num):

            i_conf_data = conf_data[i, :, :] # [num_priors, num_classes]
            i_conf_t = conf_t[i].long() #[num_priors]
            if self.use_ARM:
                num_positive = i_conf_t.gt(0).sum().float() # number of priors with IoU greater than 0.5
            else:
                num_positive = i_conf_t.size(0)
            conf_one_hot = torch.zeros(i_conf_data.shape)
            conf_one_hot.scatter_(1, i_conf_t.unsqueeze(1), 1) # label to one-hot vector, [num_priors, num_classes]
            if self.use_ARM:
                conf_one_hot[:, 0] = 0 # avoid the background
            pred_softmax = F.softmax(i_conf_data, dim=1)
            focal_loss_weight = -1*(1-pred_softmax).pow(gamma) * alpha
            i_loss = (conf_one_hot*pred_softmax.log()*focal_loss_weight).sum() / num_positive
            loss_c += i_loss
            # temp_loss_c = torch.cat((temp_loss_c, i_loss.unsqueeze(0)), 0)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        #N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        #print(N, loss_l, loss_c)
        return loss_l, loss_c
