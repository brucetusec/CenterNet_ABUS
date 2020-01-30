import os, sys, time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from models.networks.hourglass import get_large_hourglass_net
from models.loss import FocalLoss, RegL1Loss, RegLoss
from data.abus_data import AbusNpyFormat

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()

def train(args):
    trainset = AbusNpyFormat(root=root)
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    crit_hm = FocalLoss()
    crit_reg = RegL1Loss()
    crit_wh = crit_reg

    train_hist = {
        'hm_loss':[],
        'per_epoch_time':[]
    }

    heads = {
        'hm': 1,
        'wh': 3
    }
    model = get_large_hourglass_net(heads, n_stacks=1)
    # model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TODO: trim codes
    print('Training starts.')
    start_time = time.time()
    for epoch in range(args.max_epoch):
        epoch_start_time = time.time()
        for batch_idx, (data_img, data_hm, data_wh_x, data_wh_y, data_wh_z) in enumerate(trainset_loader):

            if use_cuda:
                data_img = data_img.cuda()
                data_hm = data_hm.cuda()
                data_wh_x = data_wh_x.cuda() 
                data_wh_y = data_wh_y.cuda()
                data_wh_z = data_wh_z.cuda()
                model.to(device)

            optimizer.zero_grad()

            output = model(data_img)
            hm_loss = crit_hm(output[0]['hm'], data_hm)
            train_hist['hm_loss'].append(hm_loss.item())

            total_loss = hm_loss
            total_loss.backward()

            optimizer.step()
            
            print("Epoch: [%2d] [%4d], hm_loss: %.4f" % ((epoch + 1), (batch_idx + 1), total_loss.item()))

        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        if (epoch % 2) == 0:
            model.save(str(epoch))

        print('Epoch exec time: {}'.format(time.time() - epoch_start_time))

    print("Training finished.")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-5)
    return parser.parse_args()

if __name__ == '__main__':
    if use_cuda:
        print('GPU is available.')
    args = _parse_args()
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')

    train(args)