import os
import numpy as np
from torch.utils import data
from torchvision import transforms as trnsfm

class AbusNpyFormat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, validation=False):
        self.root = root
        with open(self.root + 'annotations/old_all.txt', 'r') as f:
            lines = f.readlines()

        # TODO: 5-fold cross-validation        
        if train:
            self.imgs = lines[:int(0.8*len(lines))] 
        elif validation:
            self.imgs = lines[int(0.8*len(lines)):]
        else:
            self.imgs = lines

        # TODO: data augmentation


    def __getitem__(self, index):
        line = self.imgs[index]
        line = line.split(',', 4)

        data = np.load(self.root + 'converted_640_160_640/' + line[0].replace('/', '_'))
        true_boxes = line[-1].split(' ')

        return data, true_boxes

    def __len__(self):
        return len(self.imgs)