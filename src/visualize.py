import os, math, json, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from data.abus_data import AbusNpyFormat

def main(args, root):
    size = (int(640*args.scale),160,int(640*args.scale))
    views = {'cor', 'sag', 'tran'}

    with open(root + 'annotations/rand_all.txt', 'r') as f:
        lines = f.read().splitlines()

    line = lines[args.index]
    line = line.split(',', 4)
    boxes = line[-1].split(' ')
    boxes = list(map(lambda box: box.split(','), boxes))
    boxes = [list(map(float, box)) for box in boxes]
    gt_boxes = [{
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

    for v in views:
        img_dir = os.path.join(args.save_dir, 'pred', str(args.index), v)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if v is 'cor':
            for i in range(np.shape(img_data)[1]):
                img = Image.fromarray(img_data[:,i,:], 'L')
                img = img.convert(mode='RGB')
                draw = ImageDraw.Draw(img)
                for bx in boxes:
                    if bx['score'] < 0.1:
                        continue

                    z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot'], bx['z_top'], bx['y_bot'], bx['y_top'], bx['x_bot'], bx['x_top']
                    if int(y_bot) <= i <= int(y_top):
                        draw.rectangle(
                            [(z_bot,x_bot),(z_top,x_top)],
                            outline ="blue", width=2)
                
                for bx in gt_boxes:
                    z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], bx['x_bot']*scale[2], bx['x_top']*scale[2]

                    if int(y_bot) <= i <= int(y_top):
                        draw.rectangle(
                            [(z_bot,x_bot),(z_top,x_top)],
                            outline ="red", width=2)
                
                draw.rectangle((0, 2, 30, 12), fill='blue')
                draw.text((2,0), 'Pred', fill="white")
                draw.rectangle((0, 14, 30, 24), fill='red')
                draw.text((2,14), 'GT', fill="white")

                img.save(os.path.join(img_dir, (str(i)+'.png')))

        if v is 'sag':
            for i in range(np.shape(img_data)[0]):
                img = Image.fromarray(img_data[i,:,:], 'L')
                img = img.convert(mode='RGB')
                draw = ImageDraw.Draw(img)
                for bx in boxes:
                    if bx['score'] < 0.1:
                        continue

                    z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot'], bx['z_top'], bx['y_bot'], bx['y_top'], bx['x_bot'], bx['x_top']
                    if int(x_bot) <= i <= int(x_top):
                        draw.rectangle(
                            [(z_bot,y_bot),(z_top,y_top)],
                            outline ="blue", width=2)
                
                for bx in gt_boxes:
                    z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], bx['x_bot']*scale[2], bx['x_top']*scale[2]
                    if int(x_bot) <= i <= int(x_top):
                        draw.rectangle(
                            [(z_bot,y_bot),(z_top,y_top)],
                            outline ="red", width=2)

                draw.rectangle((0, 2, 30, 12), fill='blue')
                draw.text((2,0), 'Pred', fill="white")
                draw.rectangle((0, 14, 30, 24), fill='red')
                draw.text((2,14), 'GT', fill="white")

                img.save(os.path.join(img_dir, (str(i)+'.png')))
        
        if v is 'tran':
            for i in range(np.shape(img_data)[2]):
                img = Image.fromarray(img_data[:,:,i], 'L')
                img = img.convert(mode='RGB')
                img = img.rotate(angle=-90, expand=True)
                draw = ImageDraw.Draw(img)
                for bx in boxes:
                    if bx['score'] < 0.01:
                        continue

                    z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot'], bx['z_top'], bx['y_bot'], bx['y_top'], 640-bx['x_bot'], 640-bx['x_top']
                    if int(z_bot) <= i <= int(z_top):
                        draw.rectangle(
                            [(x_bot,y_bot),(x_top,y_top)],
                            outline ="blue", width=2)

                for bx in gt_boxes:
                    z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale[0], bx['z_top']*scale[0], bx['y_bot']*scale[1], bx['y_top']*scale[1], 640-bx['x_bot']*scale[2], 640-bx['x_top']*scale[2]
                    if int(z_bot) <= i <= int(z_top):
                        draw.rectangle(
                            [(x_bot,y_bot),(x_top,y_top)],
                            outline ="red", width=2)

                draw.rectangle((0, 2, 30, 12), fill='blue')
                draw.text((2,0), 'Pred', fill="white")
                draw.rectangle((0, 14, 30, 24), fill='red')
                draw.text((2,14), 'GT', fill="white")

                img.save(os.path.join(img_dir, (str(i)+'.png')))
    
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
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    args = _parse_args()
    main(args, root)