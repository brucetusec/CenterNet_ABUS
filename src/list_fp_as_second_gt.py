import os, argparse
import numpy as np
from utils.postprocess import pick_fp_by_dist

def main(args):
    num_npy = os.listdir(npy_dir) # dir is your directory path
    total_pass = len(num_npy)


    current_pass = 0
    with open(root + 'annotations/rand_all.txt', 'r') as f:
        lines = f.read().splitlines()

    new_lines = []
    for line in lines:
        line = line.split(',', 4)
        # Always use 640,160,640 to compute iou
        size = (640,160,640)
        scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
        pred_npy = npy_dir + line[0].replace('/', '_')
        if not os.path.exists(pred_npy):
            continue
        else:
            current_pass += 1
            print('Processing {}/{} data...'.format(current_pass, total_pass))
            if current_pass == total_pass:
                print("\n")

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
        
        ##########################################
        out_boxes = []
        box_list = np.load(pred_npy)
        for bx in box_list:
            out_boxes.append(list(bx))

        FP, fp_list = pick_fp_by_dist(out_boxes, true_box, 50, scale)
        new_lines.append(line[0]+','+line[1]+','+line[2]+','+line[3]+','+' '.join(fp_list)+'\n')
        print('Number of FPs in {}: {}'.format(line[0], FP))
        print('*************************************************')

    with open(root + 'annotations/fp_{}.txt'.format(args.fold), 'w') as f:
        f.writelines(new_lines)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fold', '-f', type=int, required=True,
        help='Which fold is the target rn?'
    )
    return parser.parse_args()


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    args = _parse_args()
    main(args)
