import torch
import torch.nn as nn
import os
import time
from models.networks.hourglass import get_large_hourglass_net

def main():
    heads = {
        'hm': 1,
        'wh': 3
    }
    model = get_large_hourglass_net(heads, n_stacks=1)
    #print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()