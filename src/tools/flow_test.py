import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
import torch
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
        'hm': 1, # 1-D Probability heat map.
        'wh': 3  # 3-D x,y,z size regression.
    }
    model = get_large_hourglass_net(heads, n_stacks=1, debug=True)
    model = model.to(device)
    
    all_data = AbusNpyFormat(root, train=False, validation=False)
    data, label = all_data.__getitem__(0)
    data = data.view(1,1,640,160,640).to(torch.float32).to(device)
    
    output = model(data)
    print('Heat map tensor:', output[0]['hm'].shape)
    print('Height-Width tensor:', output[0]['wh'].shape)
    return

if __name__=='__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data/sys_ucc/')
    main()