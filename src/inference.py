import os, sys, time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.abus_data import AbusNpyFormat
from utils.postprocess import nms, max_in_neighborhood
from models.networks.hourglass import get_large_hourglass_net
import shutil
def _get_dilated_range(coord, width, dilation=4):
    center = (dilation*coord + 2)
    return (center - width//2), (center + width//2)


def _get_topk(boxes, hm_pred, size, wh_pred, topk=10):
    dilation = (640/size[0], 160/size[1], 640/size[2])
    hmax = nms(hm_pred, 11)
    topk_scores, topk_inds = torch.topk(hmax.view(-1), topk)
    print('Top {}-{} predicted score:'.format(len(boxes)+1, len(boxes)+topk), list(map(lambda score: round(score, 3), topk_scores.tolist())))
    _z = topk_inds//(size[1]*size[0])
    _y = (topk_inds % (size[1]*size[0]))//size[2]
    _x = ((topk_inds % (size[1]*size[0])) % size[2])

    wh_pred = max_in_neighborhood(wh_pred, kernel=3)

    for i in range(topk_scores.shape[0]):
        # w0, w1, w2 should be stored in 640,160,640
        z, y, x = _z[i].item(), _y[i].item(), _x[i].item()
        w0 = wh_pred[0, 0, z, y, x].to(torch.uint8).item()
        w1 = wh_pred[0, 1, z, y, x].to(torch.uint8).item()
        w2 = wh_pred[0, 2, z, y, x].to(torch.uint8).item()

        z_bot, z_top = _get_dilated_range(z, w0, dilation=dilation[0])
        y_bot, y_top = _get_dilated_range(y, w1, dilation=dilation[1])
        x_bot, x_top = _get_dilated_range(x, w2, dilation=dilation[2])
        boxes.append([z_bot, y_bot, x_bot, z_top, y_top, x_top, round(topk_scores[i].item(), 3)])

    return boxes


def main(args, testing_mode):
    size = (160, 40, 160)

    heads = {
        'hm': 1, # 1 channel Probability heat map.
        'wh': 3  # 3 channel x,y,z size regression.
    }
    model = get_large_hourglass_net(heads, n_stacks=1, debug=False)
    model.load(chkpts_dir, args.epoch)
    model = model.to(device)
    model.eval()

    trainset = AbusNpyFormat(testing_mode, root=root, crx_valid=True, crx_fold_num=args.fold_num, crx_partition='valid')
    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data_img, hm_gt, box_gt, extra_info) in enumerate(trainset_loader):
            f_name = trainset.getName(extra_info[1])
            data_img = data_img.to(device)
            output = model(data_img)
            print('***************************')
            print('Processing: ', batch_idx+1, " / ", len(trainset.gt))
            print('Processing: ', f_name)

            wh_pred = torch.abs(output[-1]['wh'])
            hm_pred = output[-1]['hm']
            boxes = []
            # First round
            boxes = _get_topk(boxes, hm_pred, size, wh_pred, topk=50)

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
    parser.add_argument(
        '--root', '-r', type=str, required=True,
        help='folder path for data/sys_ucc/'
    )

    return parser.parse_args()


if __name__=='__main__':
    #python src/inference.py -f=0 -e=10 --root=/data/bruce/Tien-Yi/test_data/sys_ucc/
    if not torch.cuda.is_available():
        print('CUDA is unavailable, abort mission!')
        quit()
    else:
        device = torch.device('cuda:0')
    args = _parse_args()

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    root = args.root
    #python ./src/inference.py -f=0 -e=10 --root=/data/bruce/Tien-Yi/test_data/sys_ucc/
    chkpts_dir = 'checkpoints/'#os.path.join(os.path.dirname(os.path.realpath(__file__)), '')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')

    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    #[('331_F3_r2', 3)]:#[('331_F0', 0), ('331_F1', 1), ('331_F2' ,2), ('331_F3', 3), ('331_F4', 4), ]:
    #[('331_F1_r2', 1), ('331_F2_r2', 2)]:
    #[('EASON_F0', 0), ('EASON_F1', 1), ('EASON_F2', 2), ('EASON_F3', 3), ('EASON_F4', 4)]: make sure EASON = 1
    for testing_mode in [0, 1]:
        infer_list = [
            ('EASON_F1_r18', 1),
            ('EASON_F1_r19', 1),
            # ('EASON_F4_r14', 4),

            # ('EASON_F3_r3',3),
            # ('EASON_F3_r4',3),
            # ('EASON_F3_r5',3),
            # ('EASON_F3_r6',3),
            # ('EASON_F3_r7',3),
            # ('EASON_F3_r8',3),
            # ('EASON_F3_r9',3),
            # ('EASON_F3_r10',3),
            # ('EASON_F3_r11',3),
            # ('EASON_F3_r12',3),
            # ('EASON_F3_r13',3),
            # ('EASON_F3_r14',3),
            # ('EASON_F3_r15',3),
            # ('EASON_F3_r16',3),
            # ('EASON_F3_r17',3),

            # ('EASON_F1_r4',1),
            # ('EASON_F1_r5',1),
            # ('EASON_F1_r6',1),
            # ('EASON_F1_r7',1),
            # ('EASON_F1_r8',1),
            # ('EASON_F1_r9',1),
            # ('EASON_F1_r10',1),
            # ('EASON_F1_r11',1),
            # ('EASON_F1_r12',1),
            # ('EASON_F1_r13',1),
            # ('EASON_F1_r14',1),
            # ('EASON_F1_r15',1),
            # ('EASON_F1_r16',1),
            # ('EASON_F1_r17',1),
            ]
        infer_list = [(f'EASON_F1_r{_}', 1) for _ in range(18,31)]
        for fold, fold_num in infer_list:

            if testing_mode==1:
                if not os.path.exists('results/prediction/' + fold + '_testing'):
                    os.mkdir('results/prediction/' + fold + '_testing')
            else:
                if not os.path.exists('results/prediction/' + fold):
                    os.mkdir('results/prediction/' + fold)

            epoch_weights = os.listdir('checkpoints/' + fold)
            epoch_weights = [_[0:-4] for _ in epoch_weights if '.pth' in _]
            epoch_weights = [_ for _ in epoch_weights if not _ in ['initial']]
            for epoch in epoch_weights:
                args.fold_num = fold_num
                args.epoch = fold + '/' + epoch
                if testing_mode==1:
                    npy_dir = 'results/prediction/' + fold + '_testing' + '/' + epoch
                else:
                    npy_dir = 'results/prediction/' + fold + '/' + epoch

                if os.path.exists(npy_dir):
                    shutil.rmtree(npy_dir)
                assert not os.path.exists(npy_dir)
                os.mkdir(npy_dir)
                main(args, testing_mode)

