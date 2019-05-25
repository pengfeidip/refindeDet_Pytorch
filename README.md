 [PyTorch](http://pytorch.org/) implementation of [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897 ). The official and original Caffe code can be found [here](https://github.com/sfzhang15/RefineDet).

# Table of Contents
[TOC]



## Performance

Train : VOC 07+12 trainval

Test:  VOC 07 test



| Methond                     | My mAP |
| --------------------------- | ------ |
| base line                   | 81.95% |
| mixed precision             | 81.85% |
| withmixup data augmentation | 83.09% |



## Experiment Record

**2019-5-21 18:47:58**

1. Mixed precision training with [apex](https://nvidia.github.io/apex/amp.html) 
2. [Mixup](https://arxiv.org/abs/1902.04103) data augmentation with β(1.5, 1.5)

**2019-5-18 18:08:16**

1. add **csv** **model** ， **csv** **model**  is an easy way to train or evaluation，not like the **VOC which is too complicated 