import os
from tqdm import tqdm
from progress.bar import Bar
from pycocotools.coco import COCO

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from opt import Options
from pose_residual import PRN
from utils.eval import Evaluation
from utils.utils import save_options
from utils.utils import save_model, adjust_lr
from utils.data_loader import CocoDataset
import matplotlib.pyplot as plt

def main(optin):
    if not os.path.exists('checkpoint/'+optin.exp):
        os.makedirs('checkpoint/'+optin.exp)

    model = PRN(optin.node_count,optin.coeff).cuda()
    #model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=optin.lr)
    criterion = torch.nn.BCELoss().cuda()

    print (model)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    save_options(optin, os.path.join('checkpoint/' + optin.exp), model.__str__(), criterion.__str__(), optimizer.__str__())

    print ('---------Loading Coco Training Set--------')
    coco_train = COCO(os.path.join('./annotations/person_keypoints_train2017.json'))
    trainloader = DataLoader(dataset=CocoDataset(coco_train,optin),batch_size=optin.batch_size, num_workers=optin.num_workers, shuffle=True)

    bar = Bar('-->', fill='>', max=len(trainloader))

    cudnn.benchmark = True
    for epoch in range(optin.number_of_epoch):
        print ('-------------Training Epoch {}-------------'.format(epoch))
        print ('Total Step:', len(trainloader), '| Total Epoch:', optin.number_of_epoch)
        lr = adjust_lr(optimizer, epoch, optin.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        for idx, (input, label) in tqdm(enumerate(trainloader)):

            input = input.cuda().float()
            label = label.cuda().float()
            #outputs = torch.nn.parallel.data_parallel(model,input,device_ids=[12,13])
            outputs = model(input)

            #plt.subplot(221),plt.imshow(input.cpu().numpy()[0])
            #plt.title('detect')
            #plt.subplot(222),plt.imshow(outputs.cpu().numpy()[0])
            #plt.title('heatmap')
            #plt.show()

            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if idx % 200 == 0:
                bar.suffix = 'Epoch: {epoch} Total: {ttl} | ETA: {eta:} | loss:{loss}' \
                .format(ttl=bar.elapsed_td, eta=bar.eta_td, loss=loss.data, epoch=epoch)
                bar.next()

        Evaluation(model, optin)

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, checkpoint='checkpoint/' + optin.exp)

        model.train()

if __name__ == "__main__":
    option = Options().parse()
    main(option)
