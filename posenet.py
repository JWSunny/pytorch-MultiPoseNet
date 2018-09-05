# -*- coding:utf-8 -*-
# keypoint estimation subnet
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from fpn import FPN50,FPN101

class Concat(nn.Module):
    def __init__(self):
        super(Concat,self).__init__()
    def forward(self,up1,up2,up3,up4):
        return torch.cat((up1,up2,up3,up4),1)

class poseNet(nn.Module):
    def __init__(self,layers):
        super(poseNet,self).__init__()
        if layers == 101:
            self.fpn = FPN101()
        if layers == 50:
            self.fpn = FPN50()
            
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # D-layers
        # 两个3x3卷积核，把channels降到128
        self.convt1 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
        self.convt2 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
        self.convt3 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
        self.convt4 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
        self.convs1 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.convs2 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.convs3 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.convs4 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        
        self.upsample1 = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        # self.upsample4 = nn.Upsample(size=(120,120),mode='bilinear',align_corners=True)

        self.concat = Concat()
        self.conv2 = nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1)
        self.convfin = nn.Conv2d(256,17,kernel_size=1,stride=1,padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self,x):
        # c2, p3, p4, p5, p6, p7 = self.fpn(x)[1]
        p2, p3, p4, p5 = self.fpn(x)[0]
        # p2 = self._upsample_add(p3,self.latlayer4(c2))
        dt5 = self.convt1(p5)
        d5 = self.convs1(dt5)
        dt4 = self.convt2(p4)
        d4 = self.convs2(dt4)
        dt3 = self.convt3(p3)
        d3 = self.convs3(dt3)
        dt2 = self.convt4(p2)
        d2 = self.convs4(dt2)
        
        up5 = self.upsample1(d5)
        up4 = self.upsample2(d4)
        up3 = self.upsample3(d3)
        # up2 = self.upsample4(d2)

        concat = self.concat(up5,up4,up3,d2)
        smooth = F.relu(self.conv2(concat))
        predict = self.convfin(smooth)
        predict = F.upsample(predict,scale_factor=4,mode='bilinear',align_corners=True)

        return predict

