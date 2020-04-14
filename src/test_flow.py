import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.abus_data import AbusNpyFormat
from models.networks.hourglass import get_large_hourglass_net

def main(args):
    heads = {
        'hm': 1, # 1 channel Probability heat map.
        'wh': 3  # 3 channel x,y,z size regression.
    }
    model = get_large_hourglass_net(heads, n_stacks=1, debug=True)
    model = model.to(device)

    trainset = AbusNpyFormat(root=root, downsample=1)
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    for batch_idx, (data_img, hm_gt, box_gt, _) in enumerate(trainset_loader):
        data_img = data_img.to(device)
        print('Batch:', data_img.shape)
        output = model(data_img)
        print('Output length:', len(output))
        print('HM tensor:', output[-1]['hm'].shape)
        print('Box tensor:', output[-1]['wh'].shape)
        print('GT HM tensor:', hm_gt.shape)
        print('GT Box tensor:', box_gt.shape)
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