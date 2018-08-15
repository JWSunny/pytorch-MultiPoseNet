# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fpn import Bottleneck
from pose_residual import PRN

class Concat(nn.Module):
    def __init__(self):
        super(Concat,self).__init__()

    def forward(self,up1,up2,up3,up4):
        return torch.cat((up1,up2,up3,up4),0)

class MultiPoseNet(nn.Module):
    # 需要做两个fpn且共享前面resnet部分的参数
    def __init__(self,block,num_blocks):
        super(MultiPoseNet,self).__init__()
        self.in_planes = 256

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # fpn部分的子网络
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)     # 第一个bottleneck出来是d256
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
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

        # 统一concat处理
        # output_padding必须必stride和padding小
        self.upsample1 = nn.ConvTranspose2d(128,512,kernel_size=3,stride=4,padding=3,output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(128,512,kernel_size=3,stride=4,padding=3,output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(128,512,kernel_size=3,stride=4,padding=3,output_padding=1)
        self.upsample4 = nn.ConvTranspose2d(128,512,kernel_size=3,stride=4,padding=3,output_padding=1)

        self.concat = Concat()

        self.convfin = nn.Conv2d(512,17,kernel_size=1,stride=1,padding=1)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
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

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        #p6 = self.conv6(c5)
        #p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)

        # 过两个3x3卷积，到关键点回归网络的下一阶
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
        up2 = self.upsample4(d2)

        concat = self.concat(up5,up4,up3,up2)
        predict = self.convfin(concat)
        
        return predict

def MultiposeNet50():
    # [3,4,6,3] -> resnet50
    return MultiPoseNet(Bottleneck, [3,4,6,3])

def MultiposeNet101():
    # [2,4,23,3] -> resnet101
    return MultiPoseNet(Bottleneck, [2,4,23,3])

net = MultiposeNet101()
input = torch.randn(1,3,480,480)
output = net(input)
