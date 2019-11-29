import os
import numpy as np
import torch
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

        data = np.load(self.root + 'converted_640_160_640/' + line[0].replace('/', '_'))
        data = torch.from_numpy(data).unsqueeze(0).float()
        true_boxes = line[-1].split(' ')
        true_boxes = list(map(lambda box: box.split(','), true_boxes))
        true_boxes = [{
            'z_bot': box[0],
            'z_top': box[3],
            'y_bot': box[1],
            'y_top': box[4],
            'x_bot': box[2],
            'x_top': box[5],
        } for box in true_boxes]

        return data, true_boxes

    def __len__(self):
        return len(self.gt)

    def getScaleZXY(self, index, size):
        # Size should be a 3-tuple, e.g.(640,640,160)
        line = self.gt[index]
        line = line.split(',', 4)
        return (size[0]/int(line[1]),size[1]/int(line[3]),size[2]/int(line[2]))

    def getName(self, index):
        line = self.gt[index]
        line = line.split(',', 4)
        return line[0].replace('/', '_')
