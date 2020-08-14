import os, sys, argparse
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from utils.heatmap import gen_3d_heatmap, gen_3d_hw
from data.abus_data import AbusNpyFormat
np.set_printoptions(threshold=sys.maxsize)

TRANSPARENCY = .25  # Degree of transparency, 0-100%
OPACITY = int(255 * TRANSPARENCY)

def draw_slice(volume, gt, dir, label=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    min, max = torch.min(gt), torch.max(gt)
    gt = ((gt-min)/max)*255
    gt = gt.to(torch.uint8).detach().numpy()
    for i in range(np.shape(gt)[1]):
        overlay = Image.fromarray(gt[:,i,:].astype(np.uint8), 'L')
        overlay = overlay.convert(mode='RGBA')
        overlay = overlay.resize((640, 640), Image.BILINEAR)
        overlay.putalpha(160)
        draw = ImageDraw.Draw(overlay)
        if label != None:
            for bx in label:
                if int(bx['y_bot']/4) <= i <= int(bx['y_top']/4):
                    draw.rectangle([(bx['x_bot'], bx['z_bot']),(bx['x_top'], bx['z_top'])],outline ="red", width=2)

        vol = Image.fromarray(volume[:,i*4,:].astype(np.uint8), 'L')
        img = vol.convert(mode='RGBA')

        # Alpha composite these two images together to obtain the desired result.
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        img.save(os.path.join(dir ,(str(i)+'.png')))

def main(args):
    all_data = AbusNpyFormat(root, crx_valid=False, augmentation=False, include_fp=False)
    data, hm, wh, label = all_data.__getitem__(args.index)
    print('Dataset size:', all_data.__len__())
    print('Shape of data:', data.shape, hm.shape, wh.shape)

    data = data.detach().numpy()
    tmp_dir = os.path.join(os.path.dirname(__file__),'test',str(args.index),'hm')
    draw_slice(data[0], hm[0], tmp_dir, label=label[0])
    tmp_dir = os.path.join(os.path.dirname(__file__),'test',str(args.index),'wh_x')
    draw_slice(data[0], wh[2], tmp_dir, label=label[0])
    tmp_dir = os.path.join(os.path.dirname(__file__),'test',str(args.index),'wh_y')
    draw_slice(data[0], wh[1], tmp_dir, label=label[0])
    tmp_dir = os.path.join(os.path.dirname(__file__),'test',str(args.index),'wh_z')
    draw_slice(data[0], wh[0], tmp_dir, label=label[0])
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
