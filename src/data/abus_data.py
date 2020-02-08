import os
import numpy as np
import torch
from utils.heatmap import gen_3d_heatmap, gen_3d_hw
from torch.utils import data
from torchvision import transforms as trnsfm

class AbusNpyFormat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, validation=False):
        self.root = root
        with open(self.root + 'annotations/old_all.txt', 'r') as f:
            lines = f.read().splitlines()

        # TODO: 5-fold cross-validation        
        if train:
            self.gt = lines[:int(0.8*len(lines))] 
        elif validation:
            self.gt = lines[int(0.8*len(lines)):]
        else:
            self.gt = lines

        # TODO: data augmentation


    def __getitem__(self, index):
        line = self.gt[index]
        line = line.split(',', 4)

        size = (640,160,640)
        size = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))

        data = np.load(self.root + 'converted_640_160_640/' + line[0].replace('/', '_'))
        data = torch.from_numpy(data).view(1,640,160,640).to(torch.float32)
        true_boxes = line[-1].split(' ')
        true_boxes = list(map(lambda box: box.split(','), true_boxes))
        true_boxes = [list(map(int, box)) for box in true_boxes]
        true_boxes = [{
            'z_bot': box[0]*size[0],
            'z_top': box[3]*size[0],
            'z_range': box[3]*size[0] - box[0]*size[0] + 1,
            'z_center': (box[0] + box[3])*size[0] / 2,
            'y_bot': box[1]*size[1],
            'y_top': box[4]*size[1],
            'y_range': box[4]*size[1] - box[1]*size[1] + 1,
            'y_center': (box[1] + box[4])*size[1] / 2,
            'x_bot': box[2]*size[2],
            'x_top': box[5]*size[2],
            'x_range': box[5]*size[2] - box[2]*size[2] + 1,
            'x_center': (box[2] + box[5])*size[2] / 2,
        } for box in true_boxes]

        scale = 4

        hm = gen_3d_heatmap((640,160,640), true_boxes, scale)
        hm = torch.from_numpy(hm).view(1, 640//scale, 160//scale, 640//scale).to(torch.float32)

        wh_x, wh_y, wh_z = gen_3d_hw((640,160,640), true_boxes, scale)
        wh_x = torch.from_numpy(wh_x).view(1, 640//scale, 160//scale, 640//scale).to(torch.float32)
        wh_y = torch.from_numpy(wh_y).view(1, 640//scale, 160//scale, 640//scale).to(torch.float32)
        wh_z = torch.from_numpy(wh_z).view(1, 640//scale, 160//scale, 640//scale).to(torch.float32)

        return data, hm, torch.cat((wh_z, wh_y, wh_x), dim=0), index

    def __len__(self):
        return len(self.gt)

    # def getScaleZXY(self, index, size):
    #     # Size should be a 3-tuple, e.g.(640,640,160)
    #     line = self.gt[index]
    #     line = line.split(',', 4)
    #     return (size[0]/int(line[1]),size[1]/int(line[3]),size[2]/int(line[2]))

    def getName(self, index):
        line = self.gt[index]
        line = line.split(',', 4)
        return line[0].replace('/', '_')

    def getFilePath(self, index):
        line = self.gt[index]
        line = line.split(',', 4)
        return line[0]
