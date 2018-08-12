import torch
from src.fpn import FPN50

def test():
    net = FPN50()
    fms = net(torch.autograd.Variable(torch.randn(1,3,600,300)))
    for fm in fms:
        print(fm.size())