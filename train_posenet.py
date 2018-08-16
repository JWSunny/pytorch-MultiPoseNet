# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

import os
import sys
import numpy

from posenet import poseNet
# from dataloader import COCOkeypointloader

def main():
    input = torch.randn(1,3,480,480)
    model = poseNet(101).cuda()
    coco_kp_train = COCO(os.path.join('./annotations/person_keypoints_train2017.json'))
    #trainloader = COCOkeypointloader()     
    pass