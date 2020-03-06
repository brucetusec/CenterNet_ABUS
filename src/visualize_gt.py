import os, math, json, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from data.abus_data import AbusNpyFormat

def main(args, root):
    size = (int(640*args.scale),160,int(640*args.scale))

    with open(root + 'annotations/old_all.txt', 'r') as f:
        lines = f.read().splitlines()

    line = lines[args.index]
    line = line.split(',', 4)

    data = np.load(root + 'converted_{}_{}_{}/'.format(size[0], size[1], size[2]) + line[0].replace('/', '_'))
    boxes = line[-1].split(' ')
    boxes = list(map(lambda box: box.split(','), boxes))
    boxes = [list(map(float, box)) for box in boxes]
    boxes = [{
        'z_bot': box[0],
        'z_top': box[3],
        'z_range': box[3] - box[0] + 1,
        'z_center': (box[0] + box[3]) / 2,
        'y_bot': box[1],
        'y_top': box[4],
        'y_range': box[4] - box[1] + 1,
        'y_center': (box[1] + box[4]) / 2,
        'x_bot': box[2],
        'x_top': box[5],
        'x_range': box[5] - box[2] + 1,
        'x_center': (box[2] + box[5]) / 2,
    } for box in boxes]

    
    scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))

    img_dir = os.path.join(args.save_dir, 'gt', str(args.index))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i in range(np.shape(data)[1]):
        img = Image.fromarray(data[:,i,:], 'L')
        img = img.convert(mode='RGB')
        draw = ImageDraw.Draw(img)
        for bx in boxes:
            z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], bx['x_bot']*scale[2], bx['x_top']*scale[2]

            if int(y_bot) <= i <= int(y_top):
                draw.rectangle(
                    [(z_bot,x_bot),(z_top,x_top)],
                    outline ="blue", width=2)
        img.save(os.path.join(img_dir,(str(i)+'.png')))
    
    return

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir', '-s', type=str, required=True,
        help='Specify where to save visualized volume as series of images.'
    )
    parser.add_argument(
        '--index', '-i', type=int, required=True,
        help='Which image to draw.'
    )
    parser.add_argument(
        '--scale', '-r', type=float, required=False, default=1,
        help='Which image to draw.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    args = _parse_args()
    main(args, root)