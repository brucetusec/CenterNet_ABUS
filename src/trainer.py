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

def train(args):
    trainset = AbusNpyFormat(root=root)
    trainset_loader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1)

    crit_hm = FocalLoss()
    crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    crit_wh

    # TODO: trim codes
    print('Training starts.')
    start_time = time.time()
    for epoch in range(self.epoch):
        epoch_start_time = time.time()
        for batch_idx, (sdata, tdata) in enumerate(zip(self.source_loader, self.target_loader)):

            sinput, slabel = sdata
            tinput, _ = tdata
            sbatch_size = sinput.shape[0]
            tbatch_size = tinput.shape[0]
            
            sdomain_label = torch.ones(sbatch_size)
            tdomain_label = torch.zeros(tbatch_size)
            if use_cuda:
                sinput, slabel, sdomain_label = sinput.cuda(), slabel.cuda(), sdomain_label.cuda()
                tinput, tdomain_label = tinput.cuda(), tdomain_label.cuda()
                

            # source domain
            self.F_optimizer.zero_grad()
            self.C_optimizer.zero_grad()

            F_ = self.F(sinput)
            C_ = self.C(F_)
            C_loss_source = self.CE_loss(C_, slabel)
            self.train_hist['C_loss_source'].append(C_loss_source.item())


            if not self.single_domain:
                self.D_optimizer.zero_grad()
                D_ = self.D(F_)
                D_loss_source = self.BCE_loss(D_, sdomain_label)
                self.train_hist['D_loss_source'].append(D_loss_source.item())

                # target domain
                F_ = self.F(tinput)
                D_ = self.D(F_)
                D_loss_target = self.BCE_loss(D_, tdomain_label)
                self.train_hist['D_loss_target'].append(D_loss_target.item())                
                total_loss = D_loss_source + D_loss_target + C_loss_source
            else:
                total_loss = C_loss_source

            total_loss.backward()

            self.F_optimizer.step()
            self.C_optimizer.step()


            if not self.single_domain:
                self.D_optimizer.step()
                if ((batch_idx + 1) % 20) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss_s: %.4f, D_loss_t: %.4f, C_loss: %.4f" %
                        ((epoch + 1), (batch_idx + 1), self.batch_num+1, D_loss_source.item(), D_loss_target.item(), C_loss_source.item()))
            else:
                if ((batch_idx + 1) % 20) == 0:
                    print("Epoch: [%2d] [%4d/%4d], C_loss: %.4f" %
                        ((epoch + 1), (batch_idx + 1), self.batch_num+1, C_loss_source.item()))

        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        if (epoch % 2) == 0:
            self.save(epoch)

        print('Epoch exec time: {}'.format(time.time() - epoch_start_time))

    print("Training finished.")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                help='path to source images directories')
    parser.add_argument('--target_name', type=str, required=True,
                help='target domain name')
    parser.add_argument('--train_type', type=str, default='attention')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr2', type=float, default=1e-4)
    return parser.parse_args()

if __name__ == '__main__':
    if use_cuda:
        print('GPU is available.')
    args = _parse_args()
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')

    train(args)