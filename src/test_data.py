import os, sys, argparse
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from data.heatmap import gen_3d_heatmap, gen_3d_hw
from data.abus_data import AbusNpyFormat
np.set_printoptions(threshold=sys.maxsize)

def draw_slice(volume, dir, label=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    min, max = torch.min(volume), torch.max(volume)
    volume = ((volume-min)/max)*255
    volume = volume.to(torch.uint8).detach().numpy()
    for i in range(np.shape(volume)[1]):
        img = Image.fromarray(volume[:,i,:].astype(np.uint8), 'L')
        img = img.convert(mode='RGB')
        draw = ImageDraw.Draw(img)
        if label != None:
            for bx in label:
                if bx['y_center']//4== i:
                    draw.point([(int(bx['x_center']//4),int(bx['z_center']//4))],fill="red")
        img.save(os.path.join(dir ,(str(i)+'.png')))

def main(args):
    all_data = AbusNpyFormat(root, train=False, validation=False)
    data, hm, box = all_data.__getitem__(args.index)
    print('Dataset size:', all_data.__len__())
    print('Shape of data:', data.size())

    tmp_dir = os.path.join(os.path.dirname(__file__),'test','hm')
    draw_slice(hm[0], tmp_dir)
    tmp_dir = os.path.join(os.path.dirname(__file__),'test','wh_x')
    draw_slice(box[2], tmp_dir)
    tmp_dir = os.path.join(os.path.dirname(__file__),'test','wh_y')
    draw_slice(box[1], tmp_dir)
    tmp_dir = os.path.join(os.path.dirname(__file__),'test','wh_z')
    draw_slice(box[0], tmp_dir)

    return

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--index', '-i', type=int, required=True,
        help='Index of the requested data.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    args = _parse_args()
    main(args)