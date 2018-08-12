# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fpn import Bottleneck
from pose_residual import PRN

class MultiPoseNet(nn.Module):
    # 需要做两个fpn且共享前面resnet部分的参数
    def __init__(self,block,num_blocks):
        super(MultiPoseNet,self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        