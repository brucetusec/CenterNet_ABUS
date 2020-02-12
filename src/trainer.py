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
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
    min_loss = 0
    for epoch in range(args.max_epoch):
        current_loss = 0
        epoch_start_time = time.time()

        for batch_idx, (data_img, data_hm, data_wh, _) in enumerate(trainset_loader):
            if use_cuda:
                data_img = data_img.cuda()
                data_hm = data_hm.cuda()
                data_wh = data_wh.cuda()
                model.to(device)
            optimizer.zero_grad()
            output = model(data_img)
            wh_pred = torch.abs(output[0]['wh'])
            hm_loss = crit_hm(output[0]['hm'], data_hm)
            wh_loss = crit_wh(wh_pred, data_wh)

            total_loss = hm_loss + args.lambda_s*wh_loss
            current_loss += total_loss
            total_loss.backward()

            optimizer.step()
            
            print("Epoch: [%2d] [%4d], hm_loss: %.3f, wh_loss: %.3f, total_loss: %3f" \
                % ((epoch + 1), (batch_idx + 1), hm_loss.item(), wh_loss.item(), total_loss.item()))
        
        if epoch == 0 or current_loss < min_loss:
            min_loss = current_loss
            model.save(str(epoch))

        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        print('Epoch exec time: {}'.format(time.time() - epoch_start_time))

    print("Training finished.")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_s', type=float, default=0.1)
    return parser.parse_args()

if __name__ == '__main__':
    if use_cuda:
        print('GPU is available.')
    args = _parse_args()
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    chkpts_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints')
    if not os.path.exists(chkpts_dir):
        os.makedirs(chkpts_dir)

    train(args)