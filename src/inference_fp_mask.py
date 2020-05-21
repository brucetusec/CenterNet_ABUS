import os, sys, time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.abus_data import AbusNpyFormat
from utils.postprocess import nms 
from models.networks.hourglass import get_large_hourglass_net

def _get_dilated_range(coord, width, dilation=4):
    center = (dilation*coord + 2)
    return (center - width//2), (center + width//2)

def _get_topk_wipeoff(boxes, hm_pred, hm_fp,size, wh_pred, topk=10, global_max=1):
    dilation = (640/size[0], 160/size[1], 640/size[2])
    hm_pred = hm_pred * (1 - hm_fp)
    hmax = nms(hm_pred) / global_max
    topk_scores, topk_inds = torch.topk(hmax.view(-1), topk)
    print('Top {}-{} predicted score:'.format(len(boxes)+1, len(boxes)+topk), list(map(lambda score: round(score, 3), topk_scores.tolist())))
    z = topk_inds/(size[1]*size[0])
    y = (topk_inds % (size[1]*size[0]))/size[2]
    x = ((topk_inds % (size[1]*size[0])) % size[2])

    for i in range(topk_scores.shape[0]):
        # w0, w1, w2 should be stored in 640,160,640
        w0 = wh_pred[0,0,z[i],y[i],x[i]].to(torch.uint8).item()
        w1 = wh_pred[0,1,z[i],y[i],x[i]].to(torch.uint8).item()
        w2 = wh_pred[0,2,z[i],y[i],x[i]].to(torch.uint8).item()

        z_bot, z_top = _get_dilated_range(z[i], w0, dilation=dilation[0])
        y_bot, y_top = _get_dilated_range(y[i], w1, dilation=dilation[1])
        x_bot, x_top = _get_dilated_range(x[i], w2, dilation=dilation[2])
        boxes.append([z_bot.item(), y_bot.item(), x_bot.item(), z_top.item(), y_top.item(), x_top.item(), round(topk_scores[i].item(), 3)])
        # # Too lazy to refactor
        # hm_pred[0,0,z[i],y[i],x[i]] = 0
    
    return boxes

def main(args):
    size = (160, 40, 160)
 
    heads = {
        'hm': 1, # 1 channel Probability heat map.
        'wh': 3,  # 3 channel x,y,z size regression.
        'fp_hm': 1
    }
    model = get_large_hourglass_net(heads, n_stacks=1, debug=False)
    model.load(chkpts_dir, args.epoch)
    model = model.to(device)
    model.eval()

    trainset = AbusNpyFormat(root=root, crx_valid=True, crx_fold_num=args.fold_num, crx_partition='valid')
    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

    start_time = time.time()
    with torch.no_grad():
        global_max = 0
        for batch_idx, (data_img, hm_gt, box_gt, extra_info) in enumerate(trainset_loader):
            f_name = trainset.getName(extra_info[1])
            data_img = data_img.to(device)
            output = model(data_img)
            print('***************************')
            print('Processing: ', f_name)
            wh_pred = torch.abs(output[-1]['wh'])
            hm_pred = output[-1]['hm']
            hm_fp = output[-1]['fp_hm']

            boxes = []
            boxes = _get_topk_wipeoff(boxes, hm_pred, hm_fp, size, wh_pred, topk=1, global_max=1)
            if boxes[0][6] > global_max:
                global_max = boxes[0][6]

        for batch_idx, (data_img, hm_gt, box_gt, extra_info) in enumerate(trainset_loader):
            f_name = trainset.getName(extra_info[1])
            data_img = data_img.to(device)
            output = model(data_img)
            print('***************************')
            print('Processing: ', f_name)
            wh_pred = torch.abs(output[-1]['wh'])
            hm_pred = output[-1]['hm']
            hm_fp = output[-1]['fp_hm']

            boxes = []
            # First round
            boxes = _get_topk_wipeoff(boxes, hm_pred, hm_fp, size, wh_pred, topk=50, global_max=global_max)

            # Second round
            # boxes = _get_topk_wipeoff(boxes, hm_pred, hm_fp, size, wh_pred, topk=5)

            boxes = np.array(boxes, dtype=float)
            np.save(os.path.join(npy_dir, f_name), boxes)

    print("Inference finished.")
    print("Average time cost: {:.2f} sec.".format((time.time() - start_time)/trainset.__len__()))


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epoch', '-e', type=str, required=True,
        help='Which epoch of the model to be loaded?'
    )
    parser.add_argument(
        '--fold_num', '-f', type=int, default=4,
        help='Which fold serves as valid set?'
    )
    return parser.parse_args()

if __name__=='__main__':
    if not torch.cuda.is_available():
        print('CUDA is unavailable, abort mission!')
        quit()
    else:
        device = torch.device('cuda:0')
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    chkpts_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints/')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    args = _parse_args()
    main(args)