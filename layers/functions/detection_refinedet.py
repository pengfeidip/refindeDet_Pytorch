import torch
from torch.autograd import Function
from ..box_utils import decode, nms, center_size
from data import voc_refinedet as cfg
import time


class Detect_RefineDet(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh, 
                objectness_thre, keep_top_k):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = cfg[str(size)]['variance']

    def forward(self, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc_data = odm_loc_data
        conf_data = odm_conf_data

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):

            # Top 400 boxes per images(exclude the background confidence)
            conf_scores = conf_preds[i].clone()  # [num_classes, num_priors]
            values, _ = conf_scores[1:, :].max(0, keepdim=False)
            _, indexes = values.sort(0, descending=True)
            top_box_indexes = indexes[:self.top_k]
            conf_scores = conf_scores[:, top_box_indexes]

            # Decoding the selected boxes then performing NMS for every class
            default = decode(arm_loc_data[i][top_box_indexes, :], prior_data[top_box_indexes, :], self.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i][top_box_indexes, :], default, self.variance)

            for cl in range(1, self.num_classes):
                t1 = time.time()
                c_mask = conf_scores[cl].gt(self.conf_thresh) # i-th class position mask
                scores = conf_scores[cl][c_mask] # score which greater than threshold

                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4) #bounding

                # idx of highest scoring and non-overlapping boxes per class
                #print(boxes, scores)
                boxes_preserved, scores_preserved = nms(boxes, scores, self.nms_thresh, 400)
                output[i, cl, :boxes_preserved.size(0)] = \
                    torch.cat((scores_preserved.unsqueeze(1),
                               boxes_preserved), 1)

        # print(all_boxes)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1) # rank of
        # todo cannot undetstand
        # 原始代码：flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fille_(0)
        # 但是我发现这个代码是不起作用的，好像这样对tensor进行索引不可行，因此我改成了下面的代码
        # 利用masked_fill_来达成目的
        # 更改后的代码：flt.masked_fill_((rank < self.keep_top_k).unsqueeze(-1).expand_as(flt), 0)
        # 奇怪的是，
        flt.masked_fill_((rank > self.keep_top_k).unsqueeze(-1).expand_as(flt), 0)

        return output
