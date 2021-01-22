import os
import numpy as np
import torch
from utils.heatmap import gen_3d_heatmap, gen_3d_hw
from torch.utils import data

class AbusNpyFormat(data.Dataset):
    def __init__(self, testing_mode, root, crx_valid=False, crx_fold_num=0, crx_partition='train', augmentation=False, include_fp=False):
        self.root = root
        fold_list_root = '/data/bruce/CenterNet_ABUS/5_fold_list/'
        file_part = 'val' if crx_partition=='valid' else crx_partition
        if testing_mode==1 and file_part=='val':
            file_part = 'test'
        fold_list_file = fold_list_root + 'five_fold_{}_{}.txt'.format(file_part, crx_fold_num)
        with open(fold_list_file, 'r') as f:
            self.gt = f.read().splitlines()
        self.gt = [_.replace('/home/lab402/User/eason_thesis/ABUS_data/', '') for _ in self.gt]

        self.set_size = len(self.gt)
        self.aug = augmentation
        self.img_size = (640,160,640)

        print('Dataset info: Cross-validation {}, partition: {}, fold number {}, data augmentation {}'\
            .format(crx_valid, crx_partition, crx_fold_num, self.aug))


    def __getitem__(self, index):
        # 0: original, 1: flip Z, 2: flip X, 3: flip ZX
        aug_mode = index // self.set_size
        index = index % self.set_size
        line = self.gt[index]
        line = line.split(',', 4)

        # Only for L, W, H regression
        size = (640,160,640)
        gt_scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))

        scale = (4,4,4)

        # numpy array data (x,y,z) is not in the same order as gt label, which is (z,y,x)
        ori_data = np.load(self.root + 'converted_{}_{}_{}/'.format(self.img_size[0], self.img_size[1], self.img_size[2]) + line[0].replace('/', '_'))
        ori_data = torch.from_numpy(ori_data)
        ori_data = torch.transpose(ori_data, 0, 2).contiguous()
        ori_data = ori_data.view(1,self.img_size[0],self.img_size[1],self.img_size[2]).to(torch.float32)

        true_boxes = line[-1].split(' ')
        true_boxes = list(map(lambda box: box.split(','), true_boxes))
        true_boxes = [list(map(int, box)) for box in true_boxes]

        data, boxes = self._flipTensor(ori_data, true_boxes, gt_scale, aug_mode = aug_mode)
        # for box in boxes:
        #     if box['z_bot'] <= 0 or box['x_bot'] <= 0:
        #         print("A box is out of bound...")

        hm = gen_3d_heatmap(self.img_size, boxes, scale)
        hm = torch.from_numpy(hm).view(1, self.img_size[0]//scale[0], self.img_size[1]//scale[1], self.img_size[2]//scale[2]).to(torch.float32)

        wh_x, wh_y, wh_z = gen_3d_hw(self.img_size, boxes, scale)
        wh_x = torch.from_numpy(wh_x).view(1, self.img_size[0]//scale[0], self.img_size[1]//scale[1], self.img_size[2]//scale[2]).to(torch.float32)
        wh_y = torch.from_numpy(wh_y).view(1, self.img_size[0]//scale[0], self.img_size[1]//scale[1], self.img_size[2]//scale[2]).to(torch.float32)
        wh_z = torch.from_numpy(wh_z).view(1, self.img_size[0]//scale[0], self.img_size[1]//scale[1], self.img_size[2]//scale[2]).to(torch.float32)

        return data, hm, torch.cat((wh_z, wh_y, wh_x), dim=0), [boxes,index]


    def __len__(self):
        if self.aug:
            return 4*self.set_size
        else:
            return self.set_size


    def _flipTensor(self, data, true_boxes, gt_scale, aug_mode=0):
        if aug_mode == 1:
            data = torch.flip(data, [1])
            boxes = [{
                'z_bot': max(0, 640 - (box[3]*gt_scale[0])),
                'z_top': 640 - (box[0]*gt_scale[0]),
                'z_range': box[3]*gt_scale[0] - box[0]*gt_scale[0] + 1,
                'z_center': 640 - ((box[0] + box[3])*gt_scale[0] / 2),
                'y_bot': box[1]*gt_scale[1],
                'y_top': box[4]*gt_scale[1],
                'y_range': box[4]*gt_scale[1] - box[1]*gt_scale[1] + 1,
                'y_center': (box[1] + box[4])*gt_scale[1] / 2,
                'x_bot': box[2]*gt_scale[2],
                'x_top': box[5]*gt_scale[2],
                'x_range': box[5]*gt_scale[2] - box[2]*gt_scale[2] + 1,
                'x_center': (box[2] + box[5])*gt_scale[2] / 2,
            } for box in true_boxes]
        elif aug_mode == 2:
            data = torch.flip(data, [3])
            boxes = [{
                'z_bot': box[0]*gt_scale[0],
                'z_top': box[3]*gt_scale[0],
                'z_range': box[3]*gt_scale[0] - box[0]*gt_scale[0] + 1,
                'z_center': (box[0] + box[3])*gt_scale[0] / 2,
                'y_bot': box[1]*gt_scale[1],
                'y_top': box[4]*gt_scale[1],
                'y_range': box[4]*gt_scale[1] - box[1]*gt_scale[1] + 1,
                'y_center': (box[1] + box[4])*gt_scale[1] / 2,
                'x_bot': max(0, 640 - (box[5]*gt_scale[2])),
                'x_top': 640 - (box[2]*gt_scale[2]),
                'x_range': box[5]*gt_scale[2] - box[2]*gt_scale[2] + 1,
                'x_center': 640 - ((box[2] + box[5])*gt_scale[2] / 2),
            } for box in true_boxes]
        elif aug_mode == 3:
            data = torch.flip(data, [1,3])
            boxes = [{
                'z_bot': max(0, 640 - (box[3]*gt_scale[0])),
                'z_top': 640 - (box[0]*gt_scale[0]),
                'z_range': box[3]*gt_scale[0] - box[0]*gt_scale[0] + 1,
                'z_center': 640 - ((box[0] + box[3])*gt_scale[0] / 2),
                'y_bot': box[1]*gt_scale[1],
                'y_top': box[4]*gt_scale[1],
                'y_range': box[4]*gt_scale[1] - box[1]*gt_scale[1] + 1,
                'y_center': (box[1] + box[4])*gt_scale[1] / 2,
                'x_bot': max(0, 640 - (box[5]*gt_scale[2])),
                'x_top': 640 - (box[2]*gt_scale[2]),
                'x_range': box[5]*gt_scale[2] - box[2]*gt_scale[2] + 1,
                'x_center': 640 - ((box[2] + box[5])*gt_scale[2] / 2),
            } for box in true_boxes]
        else:
            boxes = [{
                'z_bot': box[0]*gt_scale[0],
                'z_top': box[3]*gt_scale[0],
                'z_range': box[3]*gt_scale[0] - box[0]*gt_scale[0] + 1,
                'z_center': (box[0] + box[3])*gt_scale[0] / 2,
                'y_bot': box[1]*gt_scale[1],
                'y_top': box[4]*gt_scale[1],
                'y_range': box[4]*gt_scale[1] - box[1]*gt_scale[1] + 1,
                'y_center': (box[1] + box[4])*gt_scale[1] / 2,
                'x_bot': box[2]*gt_scale[2],
                'x_top': box[5]*gt_scale[2],
                'x_range': box[5]*gt_scale[2] - box[2]*gt_scale[2] + 1,
                'x_center': (box[2] + box[5])*gt_scale[2] / 2,
            } for box in true_boxes]

        return data, boxes


    def getName(self, index):
        line = self.gt[index]
        line = line.split(',', 4)
        return line[0].replace('/', '_')


    def getFilePath(self, index):
        line = self.gt[index]
        line = line.split(',', 4)
        return line[0]
