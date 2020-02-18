import os, sys, time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.abus_data import AbusNpyFormat
from utils.postprocess import nms 
from models.networks.hourglass import get_large_hourglass_net

def _get_dilated_range(coord, width, scale=4):
    center = scale*coord + scale//2
    return (center - width//2), (center + width//2)

def main(args):
    scale = 4

    heads = {
        'hm': 1, # 1 channel Probability heat map.
        'wh': 3  # 3 channel x,y,z size regression.
    }
    model = get_large_hourglass_net(heads, n_stacks=1, debug=False)
    model.load(chkpts_dir, args.epoch)
    model.eval()
    model = model.to(device)

    trainset = AbusNpyFormat(root=root, crx_valid=True, crx_fold_num=4, crx_partition='valid')
    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

    start_time = time.time()
    for batch_idx, (data_img, hm_gt, box_gt, extra_info) in enumerate(trainset_loader):
        f_name = trainset.getName(extra_info[1])
        data_img = data_img.to(device)
        output = model(data_img)
        print('***************************')
        print('Processing: ', f_name)
        print('Output length:', len(output))
        wh_pred = torch.abs(output[0]['wh'])
        hmax = nms(output[0]['hm'])
        topk_scores, topk_inds = torch.topk(hmax.view(-1), 10)
        print('Top 10 predicted score:', list(map(lambda score: round(score, 3), topk_scores.tolist())))
        z = topk_inds/(40*160)
        y = (topk_inds % (40*160))/160
        x = ((topk_inds % (40*160)) % 160)

        boxes = []
        for i in range(topk_scores.shape[0]):
            w0 = wh_pred[0,0,z[i],y[i],x[i]].to(torch.uint8).item()
            w1 = wh_pred[0,1,z[i],y[i],x[i]].to(torch.uint8).item()
            w2 = wh_pred[0,2,z[i],y[i],x[i]].to(torch.uint8).item()

            z_bot, z_top = _get_dilated_range(z[i], w0, scale=scale)
            y_bot, y_top = _get_dilated_range(y[i], w1, scale=scale)
            x_bot, x_top = _get_dilated_range(x[i], w2, scale=scale)

            boxes.append([z_bot, y_bot, x_bot, z_top, y_top, x_top, round(topk_scores[i].item(), 3)])

        boxes = np.array(boxes, dtype=float)
        np.save(os.path.join(npy_dir, f_name), boxes)

    print("Inference finished.")
    print("Average time cost: {:.2f} sec.".format((time.time() - start_time)/trainset.__len__()))


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epoch', '-e', type=int, required=True,
        help='Which epoch of the model to be loaded.'
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