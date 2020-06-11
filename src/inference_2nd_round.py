import os, sys, time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.abus_data import AbusNpyFormat
from utils.postprocess import nms, max_in_neighborhood
from models.networks.hourglass import get_large_hourglass_net

def _get_dilated_range(coord, width, dilation=4):
    center = (dilation*coord + 2)
    return (center - width//2), (center + width//2)


def _get_topk(boxes, hm_pred, size, wh_pred, mask,topk=10):
    dilation = (640/size[0], 160/size[1], 640/size[2])
    hmax = nms(hm_pred, 11)
    topk_scores, topk_inds = torch.topk(hmax.view(-1), topk)
    print('Top {}-{} predicted score:'.format(len(boxes)+1, len(boxes)+topk), list(map(lambda score: round(score, 3), topk_scores.tolist())))
    _z = topk_inds/(size[1]*size[0])
    _y = (topk_inds % (size[1]*size[0]))/size[2]
    _x = ((topk_inds % (size[1]*size[0])) % size[2])

    wh_pred = max_in_neighborhood(wh_pred, kernel=3)
    mask = max_in_neighborhood(mask, kernel=3)
    mmax = torch.max(mask).item()

    for i in range(topk_scores.shape[0]):
        # w0, w1, w2 should be stored in 640,160,640
        z, y, x = _z[i].item(), _y[i].item(), _x[i].item()
        w0 = wh_pred[0, 0, z, y, x].to(torch.uint8).item()
        w1 = wh_pred[0, 1, z, y, x].to(torch.uint8).item()
        w2 = wh_pred[0, 2, z, y, x].to(torch.uint8).item()
        score_mask = mask[0, 0, z, y, x].item()
        score_mask = (1-score_mask**0.5)
        if score_mask > 0.67:
            score_mask = 1+score_mask**0.2
        else:
            score_mask = score_mask**4

        z_bot, z_top = _get_dilated_range(z, w0, dilation=dilation[0])
        y_bot, y_top = _get_dilated_range(y, w1, dilation=dilation[1])
        x_bot, x_top = _get_dilated_range(x, w2, dilation=dilation[2])
        if topk_scores[i].item() > 0.2:
            boxes.append([z_bot, y_bot, x_bot, z_top, y_top, x_top, topk_scores[i].item()])
        else:
            print('Score: {}, Mask: {}'.format(topk_scores[i].item(), score_mask))
            boxes.append([z_bot, y_bot, x_bot, z_top, y_top, x_top, topk_scores[i].item() * score_mask])
    
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
        for batch_idx, (data_img, hm_gt, box_gt, extra_info) in enumerate(trainset_loader):
            f_name = trainset.getName(extra_info[1])
            data_img = data_img.to(device)
            output = model(data_img)
            print('***************************')
            print('Processing: ', f_name)
            wh_pred = torch.abs(output[-1]['wh'])
            hm_pred = output[-1]['hm']
            fp_mask = output[-1]['fp_hm']
            boxes = []
            # First round
            boxes = _get_topk(boxes, hm_pred, size, wh_pred, fp_mask, topk=50)

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