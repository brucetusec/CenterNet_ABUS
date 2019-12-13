import os
import numpy as np
from heatmap import gaussian_radius, gaussian2D, gaussian3D
from abus_data import AbusNpyFormat

def main():
    all_data = AbusNpyFormat(root, train=False, validation=False)
    data, label = all_data.__getitem__(0)
    print('Shape of data:', data.size())
    print('Data len:', all_data.__len__())
    print('Number of boxes in data[0]:', len(label))
    print('Volumetric tensor:', data)
    print('Label:', label)

def gaussian_test():
    # for i in range(1,4):
    #     size = (i*2, i*5)
    #     r = gaussian_radius(size)

    hm = gaussian3D((5,5,5), sigma=2)
    print(hm.shape, hm)
    return hm

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data/sys_ucc/')
    # main()
    hm = gaussian_test()
    empty_3D = np.zeros((10,10,10), dtype=np.float32)
    print(empty_3D)