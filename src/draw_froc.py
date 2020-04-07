import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.postprocess import eval_precision_recall

#####################
auc_list = []

def AUC(froc_x, froc_y, x_limit):
    global auc_list
    froc_x = np.array(froc_x)
    froc_y = np.array(froc_y)

    area = np.trapz(froc_y[::-1], x=froc_x[::-1], dx=0.001)
    auc_list.append(area)
    return area


def draw_full(froc_x, froc_y, color, label, linestyle, x_limit):
    plt.plot(froc_x, froc_y, color=color, label=label, linestyle=linestyle)

def build_threshold():
    thresholds = []
    
    tmp=0.002
    for i in range(0, 74):
        thresholds.append(tmp)
        tmp += 0.002

    for i in range(0, 85):
        thresholds.append(tmp)
        tmp += 0.01
     
    return thresholds


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
            true_box_s = list(filter(lambda li: li[3]-li[0]<=20 or li[5]-li[2]<=20, true_box))
            if i == 0:
                true_num += len(true_box)
                true_small_num += len(true_box_s)
            #true_box = list(filter(lambda li: li[3]-li[0]>20 and li[5]-li[2]>20, ground_true_box))
            file_name = line[0]
            file_table.append(file_name)
            
            ##########################################
            out_boxes = []
            box_list = np.load(pred_npy)
            for bx in box_list:
                if bx[6] >= score_hit_thre:
                    out_boxes.append(list(bx))

            pred_num.append(len(out_boxes))

            TP, FP, FN, hits_index, hits_iou, hits_score = eval_precision_recall(
                out_boxes, true_box, 0.25, scale)

            TP_IOU_1, FP_IOU_1, FN_IOU_1, hits_index_IOU_1, hits_iou_IOU_1, hits_score_IOU_1 = eval_precision_recall(
                out_boxes, true_box, 0.1, scale)

            TP_table.append(TP)
            FP_table.append(FP)
            FN_table.append(FN)

            TP_table_IOU_1.append(TP_IOU_1)
            FP_table_IOU_1.append(FP_IOU_1)
            FN_table_IOU_1.append(FN_IOU_1)

            ##########################################
            # Small tumor
            out_boxes_s = []

            for bx in box_list:
                if bx[6] >= score_hit_thre and (bx[3]-bx[0]<=20 or bx[5]-bx[2]<=20):
                    out_boxes_s.append(list(bx))

            pred_small_num.append(len(out_boxes_s))

            TP_s, FP_s, FN_s, hits_index_s, hits_iou_s, hits_score_s = eval_precision_recall(
                out_boxes_s, true_box_s, 0.25, scale)

            TP_IOU_1_s, FP_IOU_1_s, FN_IOU_1_s, hits_index_IOU_1_s, hits_iou_IOU_1_s, hits_score_IOU_1_s = eval_precision_recall(
                out_boxes_s, true_box_s, 0.1, scale)

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

        if sensitivity_IOU_1 > 0.125:
            PERF_per_thre.append([
                score_hit_thre,
                total_pass,
                sensitivity,
                precision,
                sum_FP/total_pass,
                sensitivity_IOU_1,
                precision_IOU_1,
                sum_FP_IOU_1/total_pass])

            PERF_per_thre_s.append([
                score_hit_thre,
                total_pass,
                sensitivity_s,
                precision_s,
                sum_FP_s/total_pass,
                sensitivity_IOU_1_s,
                precision_IOU_1_s,
                sum_FP_IOU_1_s/total_pass])

    print('Small/All tumors: {}/{}'.format(true_small_num, true_num))

    data = np.array(PERF_per_thre)
    data_s = np.array(PERF_per_thre_s)
    # np.save(data_save_to,performnace_per_thre)

    font = {'family': 'Times New Roman',
            'size': 12}

    plt.rc('font', **font)

    if len(data) == 0:
        print('Inference result is empty.')
    else:
        draw_full(data[..., 4], data[..., 2], '#FF6D6C', 'IOU > 0.25 ', ':', 1)
        draw_full(data[..., 7], data[..., 5], '#FF0000', 'IOU > 0.10 ', '-', 1)

    if len(data_s) == 0:
        print('Inference result for small is empty.')
    else:
        draw_full(data_s[..., 4], data_s[..., 2], '#6D6CFF', 'IOU > 0.25 ', ':', 1)
        draw_full(data_s[..., 7], data_s[..., 5], '#0000FF', 'IOU > 0.10 ', '-', 1)

    axes = plt.gca()
    axes.axis([0, 10, 0, 1])
    axes.set_aspect('auto')
    x_tick = np.arange(0, 10, 1)
    y_tick = np.arange(0, 1, 0.125)
    plt.xticks(x_tick)
    plt.yticks(y_tick)
    plt.legend(loc='lower right')
    plt.grid(b=True, which='major', axis='x')
    plt.ylabel('Sensitivity')
    plt.xlabel('False Positive Per Pass')
    plt.savefig('froc_test.png')
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scale', '-s', type=int, default=1,
        help='How much were x,z downsampled?'
    )
    return parser.parse_args()


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    args = _parse_args()
    main(args)
