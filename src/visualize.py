import os, math, json, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from data.abus_data import AbusNpyFormat

def main(args, root):
    with open(root + 'annotations/rand_all.txt', 'r') as f:
        lines = f.read().splitlines()

    line = lines[args.index]
    line = line.split(',', 4)
    pred_npy = npy_dir + line[0].replace('/', '_')

    img_data = np.load(root + 'converted_640_160_640/' + line[0].replace('/', '_'))
    box_list = np.load(pred_npy)
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
        'score': box[6]
    } for box in box_list]

    img_dir = os.path.join(args.save_dir, 'pred', str(args.index))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i in range(np.shape(img_data)[1]):
        img = Image.fromarray(img_data[:,i,:], 'L')
        img = img.convert(mode='RGB')
        draw = ImageDraw.Draw(img)
        for bx in boxes:
            z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot'], bx['z_top'], bx['y_bot'], bx['y_top'], bx['x_bot'], bx['x_top']

            if int(y_bot) <= i <= int(y_top):
                draw.rectangle(
                    [(z_bot,x_bot),(z_top,x_top)],
                    outline ="red", width=2)
                draw.rectangle((z_bot,x_bot-10,z_bot+32,x_bot), fill='red')
                draw.text((z_bot+1,x_bot-10), str(bx['score']), fill="white")
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
    return parser.parse_args()

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    args = _parse_args()
    main(args, root)