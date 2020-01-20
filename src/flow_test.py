import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.abus_data import AbusNpyFormat
from models.networks.hourglass import get_large_hourglass_net

if not torch.cuda.is_available():
    print('CUDA is unavailable, abort mission!')
    quit()
else:
    device = torch.device('cuda:0')

parser = argparse.ArgumentParser()

# parser.add_argument(
#     '--save_dir', '-s', type=str, required=True,
#     help='Specify where to save visualized volume as series of images.'
# )
params = parser.parse_args()

def main():
    heads = {
        'hm': 1, # 1 channel Probability heat map.
        'wh': 3  # 3 channel x,y,z size regression.
    }
    model = get_large_hourglass_net(heads, n_stacks=1, debug=True)
    model = model.to(device)

    trainset = AbusNpyFormat(root=root)
    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    for batch_idx, (data_img, data_gt, _, _, _) in enumerate(trainset_loader):
        data_img = data_img.to(device)
        print('Batch:', data_img.shape)
        output = model(data_img)
        print('Output length:', len(output))
        print('Heat map tensor:', output[0]['hm'].shape)
        print('Height-Width tensor:', output[0]['wh'].shape)
        return

if __name__=='__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    main()