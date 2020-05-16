import os, argparse
import numpy as np

def main():
    true_num, true_small_num = 0, 0

    with open(root + 'annotations/rand_all.txt', 'r') as f:
        lines = f.read().splitlines()

    for line in lines:
        line = line.split(',', 4)
        # Always use 640,160,640 to compute iou
        size = (640,160,640)
        scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))

        boxes = line[-1].split(' ')
        boxes = list(map(lambda box: box.split(','), boxes))
        true_box = [list(map(float, box)) for box in boxes]
        true_box_s = []
        # For the npy volume (after interpolation by spacing), 4px = 1mm
        for li in true_box:
            axis = [0,0,0]
            axis[0] = (li[3] - li[0]) / 4
            axis[1] = (li[4] - li[1]) / 4
            axis[2] = (li[5] - li[2]) / 4
            if axis[0] < 10 and axis[1] < 10 and axis[2] < 10:
                true_box_s.append(li)

        true_num += len(true_box)
        true_small_num += len(true_box_s)

    print('Small/All tumors: {}/{}'.format(true_small_num, true_num))
            

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    main()
