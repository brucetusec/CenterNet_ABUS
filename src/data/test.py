import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from PIL import Image, ImageFont, ImageDraw
from heatmap import gen_3d_heatmap
from abus_data import AbusNpyFormat

def main():
    all_data = AbusNpyFormat(root, train=False, validation=False)
    data, label = all_data.__getitem__(6)
    print('Dataset size:', all_data.__len__())
    print('Shape of data:', data.size())
    print('Number of boxes in data:', len(label))
    #print('Volumetric tensor:', data)
    #print('Label:', label)

    scale=4

    hm1 = gen_3d_heatmap((640,160,640), label, scale)
    hm1 = np.transpose(hm1, (2,1,0))
    min, max = np.min(hm1), np.max(hm1)
    hm1 = ((hm1-min)/max)*255
    for i in range(np.shape(hm1)[1]):
        img = Image.fromarray(hm1[:,i,:].astype(np.uint8), 'L')
        img = img.convert(mode='RGB')
        draw = ImageDraw.Draw(img)
        for bx in label:
            if bx['y_center']//scale == i:
                draw.point([(int(bx['z_center']//scale),int(bx['x_center']//scale))],fill="red")
        img.save(os.path.join(os.path.dirname(__file__),(str(i)+'.png')))

    return

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data/sys_ucc/')
    main()