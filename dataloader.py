# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

from prn_train import opt as opt

class COCOkeypointloader(data.Dataset):
    def __init__(self,coco_train):
        self.coco = coco_train
        self.num_of_keypoints = opt.num_of_keypoints
        self.anns = self.get_anns(self.coco_train)
        # self.bbox_height = opt.coeff *28
        # self.bbox_width = opt.coeff *18

