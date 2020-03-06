import os
import numpy as np
from skimage.transform import resize

shape = (320,160,320)

def main():
    orig_dir = root + 'converted_640_160_640/'
    target_dir = root + 'converted_{}_{}_{}/'.format(shape[0], shape[1], shape[2])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(orig_dir):
        data = np.load(orig_dir + filename)
        # print(data.dtype)
        data_resized = np.uint8(resize(data, shape, preserve_range=True, anti_aliasing=True))
        # print(data_resized.dtype)
        np.save(target_dir + os.path.basename(filename), data_resized)
        

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    main()