import json
import numpy as np
import uuid

coco_anno = {}
coco_anno['info'] = {
    "description": "Breast tumor ROI cropped from 3D ABUS data.",
    "year": 2019,
    "contributor": "Sun Yat-sen University Cancer Center",
    "url": "http://www.sysucc.org.cn/"
}
coco_anno['images']=[]
coco_anno['annotations']=[]
coco_anno['categories']=[]

model_size = (640,160,640)

with open('../data/sys_ucc/annotations/old_all.txt', 'r') as f:
    for line in f:
        line = line.split(',', 4)
        scale_z, scale_y, scale_x = model_size[0]/int(line[1]), model_size[1]/int(line[2]), model_size[2]/int(line[3])

        boxes = line[-1].split(' ')
        true_box = np.array([np.array(list(map(int, box.split(',')))) for box in boxes])
        true_box[:, [0, 2]] = true_box[:, [2, 0]]
        true_box[:, [3, 5]] = true_box[:, [5, 3]]
        true_box[:,0], true_box[:,3] = true_box[:,0]*scale_x, true_box[:,3]*scale_x
        true_box[:,1], true_box[:,4] = true_box[:,1]*scale_y, true_box[:,4]*scale_y
        true_box[:,2], true_box[:,5] = true_box[:,2]*scale_z, true_box[:,5]*scale_z

        coco_anno['images'].append({
            "file_name": line[0],
            "x": int(line[1]),
            "y": int(line[2]),
            "z": int(line[3]),
            "id": uuid.uuid4().hex
        })


with open('../data/sys_ucc/annotations/all.json', 'w') as outfile:
    json.dump(coco_anno, outfile)