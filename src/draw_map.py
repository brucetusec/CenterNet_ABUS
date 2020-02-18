import os
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
    area = AUC(froc_x, froc_y, x_limit)
    plt.plot(froc_x, froc_y, color=color, label=label +
             ', Az = %.3f' % area, linestyle=linestyle)

def build_threshold():
    thresholds = []
    
    tmp=0.002
    for i in range(0, 74):
        thresholds.append(tmp)
        tmp += 0.002

    for i in range(0, 82):
        thresholds.append(tmp)
        tmp += 0.01
     
    return thresholds


def main():
    num_npy = os.listdir(npy_dir) # dir is your directory path
    total_pass = len(num_npy)
    all_thre=build_threshold()
    performnace_per_thre=[]

    for score_hit_thre in all_thre:
        print('Use threshold: {:.3f}'.format(score_hit_thre))

        TP_table, FP_table, FN_table, \
        TP_table_IOU_1, FP_table_IOU_1, FN_table_IOU_1, \
        true_num, pred_num, \
        file_table, iou_table, score_table, \
        mean_score_table, std_score_table = [], [], [], [], [], [], [], [], [], [], [], [], []

        hit_small_size_cnt = np.array([0, 0, 0])
        miss_small_size_cnt = np.array([0, 0, 0])

        hit_large_size_cnt = np.array([0, 0, 0])
        miss_large_size_cnt = np.array([0, 0, 0])


        current_pass = 0
        with open(root + 'annotations/old_all.txt', 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            line = line.split(',', 4)
            size = (640,160,640)
            scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
            pred_npy = npy_dir + line[0].replace('/', '_')
            if not os.path.exists(pred_npy):
                continue
            else:
                current_pass += 1
                print('Processing {}/{} data...'.format(current_pass, total_pass), end="\r")
                if current_pass == total_pass:
                    print("\n")

            boxes = line[-1].split(' ')
            boxes = list(map(lambda box: box.split(','), boxes))
            true_box = [list(map(float, box)) for box in boxes]

            file_name = line[0]
            file_table.append(file_name)
            true_num.append([len(true_box)])
            
            ##########################################
            out_boxes = []
            box_list = np.load(pred_npy)
            for bx in box_list:
                if bx[6] >= score_hit_thre:
                    out_boxes.append(bx)

            TP, FP, FN, hits_index, hits_iou, hits_score = eval_precision_recall(
                out_boxes, true_box, 0.25, scale)

            TP_IOU_1, FP_IOU_1, FN_IOU_1, hits_index_IOU_1, hits_iou_IOU_1, hits_score_IOU_1 = eval_precision_recall(
                out_boxes, true_box, 0.1, scale)

            # for gt_idx, box in enumerate(true_box):
            #     smallest_size=min(box[3:6]-box[0:3])
            #     largest_size=max(box[3:6]-box[0:3])
            #     smallest_tumor_type = None
            #     largest_tumor_type = None
            #     if smallest_size <= 20:
            #         smallest_tumor_type = 0
            #     elif smallest_size < 40:
            #         smallest_tumor_type = 1
            #     else:
            #         smallest_tumor_type = 2
                
            #     if largest_size <= 40:
            #         largest_tumor_type = 0
            #     elif largest_size < 60:
            #         largest_tumor_type = 1
            #     else:
            #         largest_tumor_type = 2

            #     if hits_index_ChonHua_1[gt_idx]!=-1:
            #         hit_small_size_cnt[smallest_tumor_type] += 1
            #         hit_large_size_cnt[largest_tumor_type] += 1
            #     else:
            #         miss_small_size_cnt[smallest_tumor_type] += 1
            #         miss_large_size_cnt[largest_tumor_type]+=1


            pred_num.append([len(out_boxes)])

            TP_table.append([TP])
            FP_table.append([FP])
            FN_table.append([FN])

            TP_table_IOU_1.append([TP_IOU_1])
            FP_table_IOU_1.append([FP_IOU_1])
            FN_table_IOU_1.append([FN_IOU_1])
        
        
        TP_table_sum = np.array(TP_table)
        FP_table_sum = np.array(FP_table)
        FN_table_sum = np.array(FN_table)

        TP_table_sum_IOU_1 = np.array(TP_table_IOU_1)
        FP_table_sum_IOU_1 = np.array(FP_table_IOU_1)
        FN_table_sum_IOU_1 = np.array(FN_table_IOU_1)


        sum_TP_IOU_1, sum_FP_IOU_1, sum_FN_IOU_1 = TP_table_sum_IOU_1.sum(), FP_table_sum_IOU_1.sum(), FN_table_sum_IOU_1.sum()
        sensitivity_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FN_IOU_1+1e-10)
        precision_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FP_IOU_1+1e-10)


        true_num_sum = np.array(true_num)
        pred_num_sum = np.array(pred_num)
        iou_table = np.array(iou_table)
        score_table = np.array(score_table)
        mean_score_table = np.array(mean_score_table)
        std_score_table = np.array(std_score_table)
        
        sum_TP, sum_FP, sum_FN, sum_true, sum_pred = TP_table_sum.sum(), FP_table_sum.sum(), FN_table_sum.sum(), true_num_sum.sum(), pred_num_sum.sum()
        sensitivity = sum_TP/(sum_TP+sum_FN+1e-10)
        precision = sum_TP/(sum_TP+sum_FP+1e-10)
        # sen0=hit_small_size_cnt[0]/(hit_small_size_cnt[0]+miss_small_size_cnt[0]+1e-10)
        # sen1=hit_small_size_cnt[1]/(hit_small_size_cnt[1]+miss_small_size_cnt[1]+1e-10)
        # sen2=hit_small_size_cnt[2]/(hit_small_size_cnt[2]+miss_small_size_cnt[2]+1e-10)
        
        
        # big_sen0=hit_large_size_cnt[0]/(hit_large_size_cnt[0]+miss_large_size_cnt[0]+1e-10)
        # big_sen1=hit_large_size_cnt[1]/(hit_large_size_cnt[1]+miss_large_size_cnt[1]+1e-10)
        # big_sen2=hit_large_size_cnt[2]/(hit_large_size_cnt[2]+miss_large_size_cnt[2]+1e-10)

        performnace_per_thre.append([
            score_hit_thre,
            total_pass,
            sensitivity,
            precision,
            sum_FP/total_pass,
            sensitivity_IOU_1,
            precision_IOU_1,
            sum_FP_IOU_1/total_pass])

            # sen0,sen1,sen2,big_sen0,big_sen1,big_sen2


    performnace_per_thre=np.array(performnace_per_thre)
    # np.save(data_save_to,performnace_per_thre)

    data = performnace_per_thre

    font = {'family': 'Times New Roman',
            'size': 12}

    plt.rc('font', **font)

    draw_full(data[..., 2], data[..., 3], '#FF6D6C', 'IOU > 0.25 ', ':', 1)
    draw_full(data[..., 5], data[..., 6], '#FF0000', 'IOU > 0.10 ', '-', 1)

    axes = plt.gca()
    axes.set_xlim([0, 1.01])
    x_tick = np.arange(0, 1, 0.125)
    plt.xticks(x_tick)
    axes.set_ylim([0, 1.01])
    y_tick = np.arange(0, 1, 0.125)
    plt.yticks(y_tick)


    plt.legend(loc='lower right')
    plt.ylabel('Precision')
    plt.xlabel('Sensitivity')
    plt.show()


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    main()
