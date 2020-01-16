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
    for ep in range(5, args.max_epoch):
        epoch_start_time = time.time()
        for idx, (s1_domain, s2_domain, s3_domain, t_domain) in enumerate(zip(*(source_loader+[target_loader]))):
            if use_cuda:
                s1_domain[0], s2_domain[0], s3_domain[0], t_domain[0] = \
                    s1_domain[0].cuda(), s2_domain[0].cuda(), s3_domain[0].cuda(), t_domain[0].cuda()
                s1_domain[1], s2_domain[1], s3_domain[1], t_domain[1] = \
                    s1_domain[1].cuda(), s2_domain[1].cuda(), s3_domain[1].cuda(), t_domain[1].cuda()

            loss_mtl, loss_moe, loss_adv, loss_reg = 0,0,0,0
            optimizer_e.zero_grad()
            optimizer_p.zero_grad()
            sources = [s1_domain, s2_domain, s3_domain]
            target_feature = extractor(t_domain[0])

            dom_label = torch.ones(target_feature.shape[0], dtype=torch.long).cuda()
            loss_adv += criterion(adversary(target_feature), dom_label)

            for i in range(3):
                meta_target = sources[i]
                feature = extractor(meta_target[0])
                w, preds = predictor(feature)

                actual_batch_size = meta_target[0].shape[0]
                dom_label = torch.tensor([]).long()
                dom_label = dom_label.new_full((actual_batch_size,), i).cuda()
                loss_reg += criterion_reg(w, dom_label)

                loss_mtl += criterion(preds[i], meta_target[1])

                w = w.unsqueeze(1)
                pred_multi = torch.stack(preds, dim=1)
                pred_moe = torch.bmm(w, pred_multi).squeeze(1)
                loss_moe += criterion(pred_moe, meta_target[1])

                dom_label = torch.zeros(actual_batch_size, dtype=torch.long).cuda()
                loss_adv += 0.33*criterion(adversary(feature), dom_label)

            total_loss = loss_moe + 0.5*loss_mtl + (1/(1+math.exp(ep//5)))*loss_adv + loss_reg
            total_loss.backward(retain_graph=True)
            optimizer_e.step()
            optimizer_p.step()


            if ((idx + 1) % 20) == 0:
                print("Epoch: [%2d] [%4d], Loss: %.4f" %
                    (ep, idx, total_loss.item()))
            
        _utils.save(extractor, str(ep)+'_extractor', args.train_type+'-'+args.target_name)
        _utils.save(predictor, str(ep)+'_predictor', args.train_type+'-'+args.target_name)

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