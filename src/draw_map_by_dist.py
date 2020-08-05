import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from operator import add
from utils.postprocess import centroid_distance, eval_precision_recall_by_dist, eval_precision_recall
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
    true_num, true_num_s, true_num_m, true_num_l = 0, 0, 0, 0

    for i, score_hit_thre in enumerate(all_thre):
        print('Use threshold: {:.3f}'.format(score_hit_thre))

        TP_table, FP_table, FN_table, \
        TP_table_IOU_1, FP_table_IOU_1, FN_table_IOU_1, \
        pred_num, pred_small_num, file_table, iou_table \
        = [], [], [], [], [], [], [], [], [], []
        # , score_table, mean_score_table, std_score_table
        TP_table_s, FP_table_s, FN_table_s, \
        TP_table_IOU_1_s, FP_table_IOU_1_s, FN_table_IOU_1_s = [], [], [], [], [], []
        
        TP_table_by_size_15 = [0,0,0]
        TP_table_by_size_10 = [0,0,0]

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

            # For the npy volume (after interpolation by spacing), 4px = 1mm
            # Only compute once
            if i == 0:
                for li in true_box:
                    axis = [0,0,0]
                    axis[0] = (li[3] - li[0]) // 4
                    axis[1] = (li[4] - li[1]) // 4
                    axis[2] = (li[5] - li[2]) // 4
                    max_axis = max(axis)
                    if max_axis <= 10:
                        true_num_s += 1
                    elif max_axis >= 15:
                        true_num_l += 1
                    else:
                        true_num_m += 1

                true_num += len(true_box)
                print('S/M/L/All tumors: {}/{}/{}/{}'.format(true_num_s, true_num_m, true_num_l, true_num))

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

            TP, FP, FN, hits_index, hits_iou, hits_score, TP_by_size_15 = eval_precision_recall_by_dist(
                out_boxes, true_box, 15, scale)

            TP_IOU_1, FP_IOU_1, FN_IOU_1, hits_index_IOU_1, hits_iou_IOU_1, hits_score_IOU_1, TP_by_size_10 = eval_precision_recall_by_dist(
                out_boxes, true_box, 10, scale)
            
            if FN_IOU_1 > 0 and i is 0:
                print("FN = {}: {}".format(FN_IOU_1, line[0]))

            TP_table.append(TP)
            FP_table.append(FP)
            FN_table.append(FN)

            TP_table_IOU_1.append(TP_IOU_1)
            FP_table_IOU_1.append(FP_IOU_1)
            FN_table_IOU_1.append(FN_IOU_1)

            TP_table_by_size_10 = list(map(add, TP_table_by_size_10, TP_by_size_10))
            TP_table_by_size_15 = list(map(add, TP_table_by_size_15, TP_by_size_15))
            ##########################################
        
        TP_table_sum = np.array(TP_table)
        FP_table_sum = np.array(FP_table)
        FN_table_sum = np.array(FN_table)

        TP_table_sum_IOU_1 = np.array(TP_table_IOU_1)
        FP_table_sum_IOU_1 = np.array(FP_table_IOU_1)
        FN_table_sum_IOU_1 = np.array(FN_table_IOU_1)

        sum_TP, sum_FP, sum_FN = TP_table_sum.sum(), FP_table_sum.sum(), FN_table_sum.sum()
        sensitivity = sum_TP/(sum_TP+sum_FN+1e-10)
        precision = sum_TP/(sum_TP+sum_FP+1e-10)

        sum_TP_IOU_1, sum_FP_IOU_1, sum_FN_IOU_1 = TP_table_sum_IOU_1.sum(), FP_table_sum_IOU_1.sum(), FN_table_sum_IOU_1.sum()
        sensitivity_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FN_IOU_1+1e-10)
        precision_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FP_IOU_1+1e-10)

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

        print('Threshold:{:.3f}'.format(score_hit_thre))
        print(TP_table_by_size_10, TP_table_by_size_15)
        print('Dist of Center < 15mm Sen:{:.3f}, Sen_s:{:.3f}, Sen_m:{:.3f}, Sen_l:{:.3f}, FP per pass:{:.3f}'\
            .format(sensitivity, TP_table_by_size_15[0]/true_num_s, TP_table_by_size_15[1]/true_num_m, TP_table_by_size_15[2]/true_num_l, sum_FP/total_pass))
        print('Dist of Center < 10mm Sen:{:.3f}, Sen_s:{:.3f}, Sen_m:{:.3f}, Sen_l:{:.3f}, FP per pass:{:.3f}'\
            .format(sensitivity_IOU_1, TP_table_by_size_10[0]/true_num_s, TP_table_by_size_10[1]/true_num_m, TP_table_by_size_10[2]/true_num_l, sum_FP_IOU_1/total_pass))
        print('\n')

    data = np.array(PERF_per_thre)
    data_s = np.array(PERF_per_thre_s)

    plt.rc('font',family='Times New Roman')

    if len(data) == 0:
        print('Inference result is empty.')
    else:
        draw_full(data[..., 5], data[..., 6], '#FF6D6C', 'D < 10mm', ':', 1)
        draw_full(data[..., 2], data[..., 3], '#FF0000', 'D < 15mm', '-', 1)

    # if len(data_s) == 0:
    #    print('Inference result for small is empty.')
    # else:
    #    draw_full(data_s[..., 2], data_s[..., 3], '#0000FF', 'Dist < 15mm', '-', 1)
    #    draw_full(data_s[..., 5], data_s[..., 6], '#6D6CFF', 'Dist < 10mm', ':', 1)

    # axes = plt.gca()
    # axes.set_aspect('auto')
    # axes.set_xlim(0.125, 1.0)
    plt.xlim(0.5, 1)
    x_tick = np.arange(0.5, 1, 0.1)
    x_tick = np.append(x_tick, [0.95, 0.98])
    x_tick = np.sort(x_tick)
    plt.xticks(x_tick)
    # axes.set_ylim(0.125, 1.01)
    y_tick = np.arange(0, 1.01, 0.125)
    plt.yticks(y_tick)
    # plt.grid(b=True, which='major', axis='both')
    plt.legend(loc='lower left')
    plt.ylabel('Precision')
    plt.xlabel('Sensitivity')
    plt.savefig('map_dist.png')
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold', type=float, default=0,
        help='Threshold for size filtering.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    args = _parse_args()
    main(args)
