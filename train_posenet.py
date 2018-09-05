# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
from progress.bar import Bar
from pycocotools.coco import COCO

import torch
#import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from posenetopt import Options
from posenet import poseNet
# from utils.eval import Evaluation
# from utils.utils import save_options
from utils.utils import save_model, adjust_lr
from dataloader import COCOkeypointloader
import matplotlib.pyplot as plt

def main(optin):
    if not os.path.exists('checkpoint/'+optin.exp):
        os.makedirs('checkpoint/'+optin.exp)
    model = poseNet(101).cuda()
    model.train()
    #model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=optin.lr)
    criterion = torch.nn.MSELoss().cuda()
    # print(os.path.join('./annotations/person_keypoints_train2017.json'))
    coco_train = COCO(os.path.join('./annotations/person_keypoints_train2017.json'))
    trainloader = DataLoader(dataset=COCOkeypointloader(coco_train),batch_size=optin.batch_size, num_workers=optin.num_workers, shuffle=True)

    bar = Bar('-->', fill='>', max=len(trainloader))

    for epoch in range(optin.number_of_epoch):
        print ('-------------Training Epoch {}-------------'.format(epoch))
        print ('Total Step:', len(trainloader), '| Total Epoch:', optin.number_of_epoch)
        lr = adjust_lr(optimizer, epoch, optin.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        for idx, (input, label) in tqdm(enumerate(trainloader)):

            input = input.cuda().float()
            label = label.cuda().float()
            
            outputs = model(input)

            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            print('Epoch {} : loss {}'.format(epoch,loss.data))
            #if idx % 200 == 0:
            #    bar.suffix = 'Epoch: {epoch} Total: {ttl} | ETA: {eta:} | loss:{loss}' \
            #    .format(ttl=bar.elapsed_td, eta=bar.eta_td, loss=loss.data, epoch=epoch)
            #    bar.next()
        if epoch % 5 == 0:
            torch.save(model,os.path.join('checkpoint/'+optin.exp, 'model_{}.pth'.format(epoch)))

if __name__ == "__main__":
    option = Options().parse()
    main(option)
