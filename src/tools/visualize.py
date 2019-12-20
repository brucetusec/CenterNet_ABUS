import os, math, json, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from data.abus_data import AbusNpyFormat

parser = argparse.ArgumentParser()

parser.add_argument(
    '--save_dir', '-s', type=str, required=True,
    help='Specify where to save visualized volume as series of images.'
)
params = parser.parse_args()

def main(root, id):
    data = AbusNpyFormat(root, train=False, validation=False)
    torch_vol, boxes = data.__getitem__(id)
    # Z,Y,X
    np_vol = torch_vol.numpy()
    scale_zxy = data.getScaleZXY(0,(640,640,160))
    
    img_dir = os.path.join(params.save_dir, data.getName(id))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i in range(np.shape(np_vol)[1]):
        img = Image.fromarray(np_vol[:,i,:], 'L')
        img = img.convert(mode='RGB')
        draw = ImageDraw.Draw(img)
        for bx in boxes:
            z_bot, z_top, y_bot, y_top, x_bot, x_top =bx['z_bot']*scale_zxy[0], bx['z_top']*scale_zxy[0], bx['y_bot']*scale_zxy[2], bx['y_top']*scale_zxy[2], bx['x_bot']*scale_zxy[1], bx['x_top']*scale_zxy[1]

            if int(y_bot) <= i <= int(y_top):
                draw.rectangle(
                    [(z_bot,x_bot),(z_top,x_top)],
                    outline ="red", width=2)
        img.save(os.path.join(img_dir,(str(i)+'.png')))
    
    return

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data/sys_ucc/')
    main(root, 6)