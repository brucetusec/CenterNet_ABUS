import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.abus_data import AbusNpyFormat
from models.networks.hourglass import get_large_hourglass_net
from models.loss import FocalLoss, RegL1Loss
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()

def main(args):
    heads = {
        'hm': 1, # 1 channel Probability heat map.
        'wh': 3  # 3 channel x,y,z size regression.
    }
    model = get_large_hourglass_net(heads, n_stacks=1, debug=True)

    trainset = AbusNpyFormat(root=root)
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    crit_hm = FocalLoss()
    crit_reg = RegL1Loss()
    crit_wh = crit_reg

    for batch_idx, (data_img, data_hm, data_wh, _) in enumerate(trainset_loader):
        if use_cuda:
            data_img = data_img.cuda()
            data_hm = data_hm.cuda()
            data_wh = data_wh.cuda()
            model.to(device)

        output = model(data_img)

        wh_pred = torch.abs(output[-1]['wh'])
        hm_loss = crit_hm(output[-1]['hm'], data_hm)
        wh_loss = 100*crit_wh(wh_pred, data_wh)

        print("hm_loss: %.3f, wh_loss: %.3f" \
                % (hm_loss.item(), wh_loss.item()))
        return

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', '-s', type=int, required=True,
        help='Specify batch size.'
    )
    return parser.parse_args()

if __name__=='__main__':
    if not torch.cuda.is_available():
        print('CUDA is unavailable, abort mission!')
        quit()
    else:
        device = torch.device('cuda:0')
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    args = _parse_args()
    main(args)