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

1. add **csv format** .  **csv format **  is an easy way to define your own dataset，not like the **VOC format** which is too complicated .

   train,val test with **csv format**  like the following

    `im_path, xmin, ymin, xmax, ymax, class_name`

   eg:

   ```
   /root/data/07+12_trainval/000005.jpg,263,211,324,339,chair
   /root/data/07+12_trainval/000005.jpg,165,264,253,372,chair
   /root/data/07+12_trainval/000005.jpg,241,194,295,299,chair
   /root/data/07+12_trainval/007212.jpg,143,9,445,331,dog
   /root/data/07+12_trainval/002657.jpg,212,198,288,264,horse
   /root/data/07+12_trainval/002657.jpg,268,353,296,375,person
   ```

   Also , need a `.csv` file named `classes.csv` 

 