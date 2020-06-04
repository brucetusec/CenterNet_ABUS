import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.postprocess import centroid_distance, eval_precision_recall_by_dist
from utils.misc import draw_full, build_threshold

def check_boundary(ct):
    y = (ct[1] > 130 or ct[1] < 5)
    z = (ct[0] > 600 or ct[0] < 40)
    x = (ct[2] > 600 or ct[2] < 40)
    return y or (z and x)


def check_size(axis, size):
    return axis[0]*axis[1]*axis[2] > size


def main(args):
    num_npy = os.listdir(npy_dir) # dir is your directory path
    total_pass = len(num_npy)
    all_thre=build_threshold()
    PERF_per_thre=[]
    PERF_per_thre_s=[]
    true_num, true_small_num = 0, 0

    for i, score_hit_thre in enumerate(all_thre):
        print('Use threshold: {:.3f}'.format(score_hit_thre))

        TP_table, FP_table, FN_table, \
        TP_table_IOU_1, FP_table_IOU_1, FN_table_IOU_1, \
        pred_num, pred_small_num, file_table, iou_table \
        = [], [], [], [], [], [], [], [], [], []
        # , score_table, mean_score_table, std_score_table
        TP_table_s, FP_table_s, FN_table_s, \
        TP_table_IOU_1_s, FP_table_IOU_1_s, FN_table_IOU_1_s = [], [], [], [], [], []

        current_pass = 0
        with open(root + 'annotations/rand_all.txt', 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            line = line.split(',', 4)
            # Always use 640,160,640 to compute iou
            size = (640,160,640)
            scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
            pred_npy = npy_dir + line[0].replace('/', '_')
            if not os.path.exists(pred_npy):
                continue
            else:
                current_pass += 1
                print('Processing {}/{} data...'.format(current_pass, total_pass), end='\r')
                if current_pass == total_pass:
                    print("\n")

            boxes = line[-1].split(' ')
            boxes = list(map(lambda box: box.split(','), boxes))
            true_box = [list(map(float, box)) for box in boxes]
            true_box_s = []
            # For the npy volume (after interpolation by spacing), 4px = 1mm
            for li in true_box:
                axis = [0,0,0]
                axis[0] = (li[3] - li[0]) / 4
                axis[1] = (li[4] - li[1]) / 4
                axis[2] = (li[5] - li[2]) / 4
                if axis[0] < 10 and axis[1] < 10 and axis[2] < 10:
                    true_box_s.append(li)

            if i == 0:
                true_num += len(true_box)
                true_small_num += len(true_box_s)

            file_name = line[0]
            file_table.append(file_name)
            
            ##########################################
            out_boxes = []
            box_list = np.load(pred_npy)
            for bx in box_list:
                axis = [0,0,0]
                axis[0] = (bx[3] - bx[0]) / scale[0] / 4
                axis[1] = (bx[4] - bx[1]) / scale[1] / 4
                axis[2] = (bx[5] - bx[2]) / scale[2] / 4
                ct = [0,0,0]
                ct[0] = (bx[3] + bx[0]) / 2
                ct[1] = (bx[4] + bx[1]) / 2
                ct[2] = (bx[5] + bx[2]) / 2
                if bx[6] >= score_hit_thre and (not check_boundary(ct)) and check_size(axis, args.threshold):
                    out_boxes.append(list(bx))

            pred_num.append(len(out_boxes))

            TP, FP, FN, hits_index, hits_iou, hits_score = eval_precision_recall_by_dist(
                out_boxes, true_box, 15, scale)

            TP_IOU_1, FP_IOU_1, FN_IOU_1, hits_index_IOU_1, hits_iou_IOU_1, hits_score_IOU_1 = eval_precision_recall_by_dist(
                out_boxes, true_box, 10, scale)
            
            if FN_IOU_1 > 0 and i is 0:
                print("FN = {}: {}".format(FN_IOU_1, line[0]))

            TP_table.append(TP)
            FP_table.append(FP)
            FN_table.append(FN)

            TP_table_IOU_1.append(TP_IOU_1)
            FP_table_IOU_1.append(FP_IOU_1)
            FN_table_IOU_1.append(FN_IOU_1)

            ##########################################
            # Small tumor

            TP_s, FP_s, FN_s, hits_index_s, hits_iou_s, hits_score_s = eval_precision_recall_by_dist(
                out_boxes, true_box_s, 15, scale)

            TP_IOU_1_s, FP_IOU_1_s, FN_IOU_1_s, hits_index_IOU_1_s, hits_iou_IOU_1_s, hits_score_IOU_1_s = eval_precision_recall_by_dist(
                out_boxes, true_box_s, 10, scale)

            TP_table_s.append(TP_s)
            FP_table_s.append(FP_s)
            FN_table_s.append(FN_s)

            TP_table_IOU_1_s.append(TP_IOU_1_s)
            FP_table_IOU_1_s.append(FP_IOU_1_s)
            FN_table_IOU_1_s.append(FN_IOU_1_s)
        
        TP_table_sum = np.array(TP_table)
        FP_table_sum = np.array(FP_table)
        FN_table_sum = np.array(FN_table)

        TP_table_sum_IOU_1 = np.array(TP_table_IOU_1)
        FP_table_sum_IOU_1 = np.array(FP_table_IOU_1)
        FN_table_sum_IOU_1 = np.array(FN_table_IOU_1)

        TP_table_sum_s = np.array(TP_table_s)
        FP_table_sum_s = np.array(FP_table_s)
        FN_table_sum_s = np.array(FN_table_s)

        TP_table_sum_IOU_1_s = np.array(TP_table_IOU_1_s)
        FP_table_sum_IOU_1_s = np.array(FP_table_IOU_1_s)
        FN_table_sum_IOU_1_s = np.array(FN_table_IOU_1_s)

        sum_TP, sum_FP, sum_FN = TP_table_sum.sum(), FP_table_sum.sum(), FN_table_sum.sum()
        sensitivity = sum_TP/(sum_TP+sum_FN+1e-10)
        precision = sum_TP/(sum_TP+sum_FP+1e-10)

        sum_TP_IOU_1, sum_FP_IOU_1, sum_FN_IOU_1 = TP_table_sum_IOU_1.sum(), FP_table_sum_IOU_1.sum(), FN_table_sum_IOU_1.sum()
        sensitivity_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FN_IOU_1+1e-10)
        precision_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FP_IOU_1+1e-10)

        sum_TP_s, sum_FP_s, sum_FN_s = TP_table_sum_s.sum(), FP_table_sum_s.sum(), FN_table_sum_s.sum()
        sensitivity_s = sum_TP_s/(sum_TP_s+sum_FN_s+1e-10)
        precision_s = sum_TP_s/(sum_TP_s+sum_FP_s+1e-10)

        sum_TP_IOU_1_s, sum_FP_IOU_1_s, sum_FN_IOU_1_s = TP_table_sum_IOU_1_s.sum(), FP_table_sum_IOU_1_s.sum(), FN_table_sum_IOU_1_s.sum()
        sensitivity_IOU_1_s = sum_TP_IOU_1_s/(sum_TP_IOU_1_s+sum_FN_IOU_1_s+1e-10)
        precision_IOU_1_s = sum_TP_IOU_1_s/(sum_TP_IOU_1_s+sum_FP_IOU_1_s+1e-10)

        if sensitivity > 0.125:
            PERF_per_thre.append([
                score_hit_thre,
                total_pass,
                sensitivity,
                precision,
                sum_FP/total_pass,
                sensitivity_IOU_1,
                precision_IOU_1,
                sum_FP_IOU_1/total_pass])

        if sensitivity_s > 0.125:
            PERF_per_thre_s.append([
                score_hit_thre,
                total_pass,
                sensitivity_s,
                precision_s,
                sum_FP_s/total_pass,
                sensitivity_IOU_1_s,
                precision_IOU_1_s,
                sum_FP_IOU_1_s/total_pass])

        print('Threshold:{:.3f}'.format(score_hit_thre))
        print('Dist of Center < 15mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity, precision, sum_FP/total_pass))
        print('Dist of Center < 10mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity_IOU_1, precision_IOU_1, sum_FP_IOU_1/total_pass))
        print('\n')

    print('Small/All tumors: {}/{}'.format(true_small_num, true_num))

    data = np.array(PERF_per_thre)
    data_s = np.array(PERF_per_thre_s)

    font = {'family': 'Times New Roman',
            'size': 9}

    plt.rc('font', **font)

    if len(data) == 0:
        print('Inference result is empty.')
    else:
        draw_full(data[..., 2], data[..., 3], '#FF0000', 'Dist < 15mm', '-', 1)
        draw_full(data[..., 5], data[..., 6], '#FF6D6C', 'Dist < 10mm', ':', 1)

    if len(data_s) == 0:
       print('Inference result for small is empty.')
    else:
       draw_full(data_s[..., 2], data_s[..., 3], '#0000FF', 'Dist < 15mm', '-', 1)
       draw_full(data_s[..., 5], data_s[..., 6], '#6D6CFF', 'Dist < 10mm', ':', 1)

    axes = plt.gca()
    axes.set_aspect('auto')
    axes.set_xlim(0.125, 1.0)
    x_tick = np.arange(0, 1, 0.1)
    plt.xticks(x_tick)
    axes.set_ylim(0.125, 1.01)
    y_tick = np.arange(0, 1.01, 0.125)
    plt.yticks(y_tick)
    plt.grid(b=True, which='major', axis='x')
    plt.legend(loc='lower left')
    plt.ylabel('Precision')
    plt.xlabel('Sensitivity')
    plt.savefig('map_dist.png')
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold', '-t', type=float, default=1,
        help='Threshold for size filtering.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    args = _parse_args()
    main(args)
