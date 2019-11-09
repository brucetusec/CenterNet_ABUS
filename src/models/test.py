import torch
import torch.nn as nn
import os
import time
from networks.hourglass import get_large_hourglass_net

def main():
    heads = {
        'hm': 2,
        'wh': 2
    }
    model = get_large_hourglass_net(heads)
    print(model)



if __name__ == '__main__':
    main()