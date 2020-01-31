import os, math, json, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from data.abus_data import AbusNpyFormat

def main(args, root, id):
    with open(root + 'annotations/old_all.txt', 'r') as f:
        lines = f.read().splitlines()

    line = lines[id]
    line = line.split(',', 4)

    data = np.load(root + 'converted_640_160_640/' + line[0].replace('/', '_'))
    true_boxes = line[-1].split(' ')
    true_boxes = list(map(lambda box: box.split(','), true_boxes))
    true_boxes = [list(map(int, box)) for box in true_boxes]
    true_boxes = [{
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
    } for box in true_boxes]

    size = (640,160,640)
    scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))

    img_dir = os.path.join(args.save_dir, str(id))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i in range(np.shape(data)[1]):
        img = Image.fromarray(data[:,i,:], 'L')
        img = img.convert(mode='RGB')
        draw = ImageDraw.Draw(img)
        for bx in true_boxes:
            z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], bx['x_bot']*scale[2], bx['x_top']*scale[2]

            if int(y_bot) <= i <= int(y_top):
                draw.rectangle(
                    [(z_bot,x_bot),(z_top,x_top)],
                    outline ="red", width=2)
        img.save(os.path.join(img_dir,(str(i)+'.png')))
    
    return

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir', '-s', type=str, required=True,
        help='Specify where to save visualized volume as series of images.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    args = _parse_args()
    main(args, root, 108)