import torch
from models.networks.hourglass import get_large_hourglass_net

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main():
    heads = {
        'hm': 1,
        'wh': 3
    }
    model = get_large_hourglass_net(heads, n_stacks=2)
    model.to(device)
    mem_cost = torch.cuda.memory_allocated()
    print('Memory cost: {:.3f} GB.'.format(mem_cost/(1024*1024*1024)))


if __name__ == '__main__':
    if use_cuda:
        print('GPU is available.')
    main()