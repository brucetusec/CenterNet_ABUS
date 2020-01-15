import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from PIL import Image, ImageFont, ImageDraw
from heatmap import gen_3d_heatmap, gen_3d_hw
from abus_data import AbusNpyFormat

def draw_slice(volume, dir, label=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    min, max = np.min(volume), np.max(volume)
    volume = ((volume-min)/max)*255
    for i in range(np.shape(volume)[1]):
        img = Image.fromarray(volume[:,i,:].astype(np.uint8), 'L')
        img = img.convert(mode='RGB')
        draw = ImageDraw.Draw(img)
        if label != None:
            for bx in label:
                if bx['y_center']//4== i:
                    draw.point([(int(bx['x_center']//4),int(bx['z_center']//4))],fill="red")
        img.save(os.path.join(dir ,(str(i)+'.png')))

def main():
    all_data = AbusNpyFormat(root, train=False, validation=False)
    data, label = all_data.__getitem__(6)
    print('Dataset size:', all_data.__len__())
    print('Shape of data:', data.size())
    print('Number of boxes in data:', len(label))

    scale=4

    hm = gen_3d_heatmap((640,160,640), label, scale)
    #hm = np.transpose(hm, (2,1,0))
    tmp_dir = os.path.join(os.path.dirname(__file__),'test','hm')
    draw_slice(hm, tmp_dir, label)

    hw_x, hw_y, hw_z = gen_3d_hw((640,160,640), label, scale)
    tmp_dir = os.path.join(os.path.dirname(__file__),'test','hw_x')
    draw_slice(hw_x, tmp_dir)
    tmp_dir = os.path.join(os.path.dirname(__file__),'test','hw_y')
    draw_slice(hw_y, tmp_dir)
    tmp_dir = os.path.join(os.path.dirname(__file__),'test','hw_z')
    draw_slice(hw_z, tmp_dir)

    return

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data/sys_ucc/')

    main()