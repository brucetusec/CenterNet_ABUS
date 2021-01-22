import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.postprocess import centroid_distance, eval_precision_recall_by_dist
from utils.misc import draw_full, build_threshold, AUC

def check_boundary(ct):
    y = (ct[1] > 130 or ct[1] < 5)
    z = (ct[0] > 600 or ct[0] < 40)
    x = (ct[2] > 600 or ct[2] < 40)
    return y or (z and x)


def check_size(axis, size):
    return axis[0]*axis[1]*axis[2] > size

def interpolate_FROC_data(froc_x, froc_y, max_fp):
        y_interpolate = 0
        take_i = 0
        for i in range(len(froc_x)):
            FP = froc_x[i]
            if FP<=max_fp:
                take_i = i
                x1 = FP
                y1 = froc_y[i]
                if i>0:
                    x2 = froc_x[i-1]
                    y2 = froc_y[i-1]

                    x_interpolate = max_fp
                    y_interpolate = (y1 * (x2-x_interpolate) + y2 * (x_interpolate-x1)) / (x2-x1)
                else:
                    #if no data point for FP > 8
                    #use sensitivity at FP = FP_small
                    y_interpolate = y1
                print("take i = ", i, " FP = ", int(FP*100)/100)
                print("interpolate sen = ", y_interpolate, " for FP=", max_fp)
                break
            else:
                print("skip i = ", i, " FP = ", int(FP*100)/100)
        froc_x = froc_x[take_i:]
        froc_y = froc_y[take_i:]

        if not froc_x[0]==max_fp:
            froc_x = np.insert(froc_x, 0, max_fp)
            froc_y = np.insert(froc_y, 0, y_interpolate)
        return froc_x, froc_y

def froc_take_max(froc_x, froc_y):
    froc_x_tmp = []
    froc_y_tmp = []
    for i in range(len(froc_x)):
        if i==0 or froc_x_tmp[-1] > froc_x[i]:
            froc_x_tmp.append(froc_x[i])
            froc_y_tmp.append(froc_y[i])
    froc_x = np.array(froc_x_tmp)
    froc_y = np.array(froc_y_tmp)
    return froc_x, froc_y

def main(args, froc_npys, exp_names):
    num_npy = os.listdir(npy_dir) # dir is your directory path
    total_pass = len(num_npy)
    all_thre=build_threshold()
    PERF_per_thre=[]
    PERF_per_thre_s=[]
    true_num, true_small_num = 0, 0

    plt.rc('font',family='Times New Roman', weight='bold')
    colors = [['#000000', '#6C6D6C'], ['#FF6D6C', '#FF0000'], ['#6D6CFF', '#0000FF'], ['#6DFF6C', '#00FF00'], ]
    data_dict = {}
    for i, (froc_npy, exp_name) in enumerate(zip(froc_npys, exp_names)):
        color, color_s = colors[i]
        data = np.load(froc_npy)
        if len(data) == 0:
            print('Inference result is empty.')
        else:
            froc_x, froc_y = interpolate_FROC_data(data[..., 7], data[..., 5], max_fp=8)
            froc_x, froc_y = froc_take_max(froc_x, froc_y)
            draw_full(froc_x, froc_y, color_s, exp_name + ' D < 10 mm', ':', 1, True)
            area_small = AUC(froc_x, froc_y, normalize=True)

            froc_x, froc_y = interpolate_FROC_data(data[..., 4], data[..., 2], max_fp=8)
            froc_x, froc_y = froc_take_max(froc_x, froc_y)
            draw_full(froc_x, froc_y, color, exp_name + ' D < 15 mm', '-', 1, True)
            area_big = AUC(froc_x, froc_y, normalize=True)

            max_sen_FP, max_sen = froc_take_max(data[..., 7], data[..., 5])
            max_sen_FP, max_sen = max_sen_FP.max(), max_sen.max()
            data_dict[exp_name] = {
                'max_sen_FP':max_sen_FP, 'max_sen':max_sen
            }
            for target_sen in [70, 75, 80, 85, 90, 93, 95, 97]:
                froc_y, froc_x = interpolate_FROC_data(data[..., 5]*100, data[..., 7], max_fp=target_sen)
                #FP_at_target
                if data[..., 5].max() < target_sen*0.01:
                    data_dict[exp_name]['FP_at_sen_' + str(target_sen)] = -1
                else:
                    data_dict[exp_name]['FP_at_sen_' + str(target_sen)] = froc_x[0]

            for target_FP in [0.125, 0.25, 0.5, 1, 2, 4, 8, 32, 128]:
                froc_x, froc_y = interpolate_FROC_data(data[..., 4], data[..., 2], max_fp=target_FP)
                #FP_at_target
                data_dict[exp_name]['sen_at_FP_' + str(target_FP)] = froc_y[0]


    for exp_name in data_dict.keys():
        max_sen_FP, max_sen = data_dict[exp_name]['max_sen_FP'], data_dict[exp_name]['max_sen']
        print('{}, {:.2f}, {:.4f}'.format(exp_name, max_sen_FP, max_sen))

        print('FP compare')
        for target_sen in [70, 75, 80, 85, 90, 93, 95, 97]:
            print('{},    {:.2f}'.format(target_sen, data_dict[exp_name]['FP_at_sen_' + str(target_sen)]))

        print('CPM') #Competition Performance Metric
        for target_FP in [0.125, 0.25, 0.5, 1, 2, 4, 8, 32, 128]:
            print('{},    {:.3f}'.format(target_FP, data_dict[exp_name]['sen_at_FP_' + str(target_FP)]))

    # axes = plt.gca()
    # axes.axis([0, 10, 0.5, 1])
    # axes.set_aspect('auto')
    x_tick = np.arange(0, 10, 2).astype(np.float)
    plt.xticks(x_tick)
    plt.ylim(0.5, 1)
    y_tick = np.arange(0.5, 1, 0.05)
    y_tick = np.append(y_tick, 0.98)
    y_tick = np.sort(y_tick)
    plt.yticks(y_tick)
    plt.legend(loc='lower right')
    # plt.grid(b=True, which='major', axis='x')
    plt.ylabel('Sensitivity')
    plt.xlabel('False Positive Per Pass')
    plt.savefig('froc_test.png')
    return 0

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold', type=float, default=0,
        help='Threshold for size filtering.'
    )
    parser.add_argument(
        '--root', '-r', type=str, required=True,
        help='folder path for data/sys_ucc/'
    )
    parser.add_argument(
        '--data_npy', type=str, required=True, action='append',
        help='npy file path for froc data'
    )
    parser.add_argument(
        '--exp_name', type=str, required=True, action='append',
        help='the label of corresponding data_npy'
    )
    return parser.parse_args()


def help_get_max_in_log():
    a = [
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0015000.pth', 0.4963698332879907, 0.5489462676957733],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0017500.pth', 0.5686607142852019, 0.6171251608746049],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0020000.pth', 0.6598434879678935, 0.6987625540074601],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0022500.pth', 0.7102960102953704, 0.738071106820442],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0025000.pth', 0.7642213642206758, 0.7970881595874416],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0027500.pth', 0.7163288288281835, 0.7491795366788617],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0030000.pth', 0.7180743243236773, 0.7519958269951496],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0032500.pth', 0.7306555279762983, 0.7631676319169445],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0035000.pth', 0.703266659516026, 0.7342058444993006],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0037500.pth', 0.7144224581718146, 0.7338642213635602],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0040000.pth', 0.7081965894459513, 0.7358671171164541],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0042500.pth', 0.6898085585579372, 0.709274453023814],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0045000.pth', 0.7118476088864412, 0.7421576576569892],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0047500.pth', 0.7119932432426018, 0.7330759330752726],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0050000.pth', 0.6750482625476544, 0.7086148648642264],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0052500.pth', 0.7379906692400043, 0.7530166103596819],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0055000.pth', 0.6954070141563876, 0.72043918918854],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0057500.pth', 0.7005389317883007, 0.7224340411833902],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0060000.pth', 0.7113095238088829, 0.7395994208487545],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0062500.pth', 0.7479407979401242, 0.7737853925346954],
['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0065000.pth', 0.7200005926973124, 0.7604810167303315],

    ]
    a = np.array(a)

    print("max ..., 1 : ", a[..., 1].astype(np.float).max(), " at ", a[a[..., 1].argmax()][0])
    print("max ..., 2 : ", a[..., 2].astype(np.float).max(), " at ", a[a[..., 2].argmax()][0])


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')

    npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    npy_format = npy_dir + '{}'

    #npy_dir = '/data/Hiola/YOLOv4-pytorch/data/pred_result/evaluate/'
    #npy_format = npy_dir + '{}_0.npy'
    #FCOS
    #npy_dir = '/data/bruce/FCOS_3D/FCOS/debug_evaluate/'
    #npy_format = npy_dir + '{}_0.npy'

    args = _parse_args()
    root = args.root
    #help_get_max_in_log()
    main(args, args.data_npy, args.exp_name)


    ##2
    """
    test_list = [

        "fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd0/model_0012500_pth",
        "fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1/model_0025000_pth",
        "fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd2/model_0032500_pth",
        "fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd3/model_0060000_pth",
        "fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd4/model_0015000_pth",
    ]

    """

    #max ..., 1 :  0.8517025089596575  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd0/model_0020000.pth
    #max ..., 1 :  0.8656370656362858  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1/model_0025000.pth
    #max ..., 1 :  0.7826428766741317  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd2/model_0015000.pth
    #max ..., 1 :  0.8379989495790103  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd3/model_0060000.pth
    #max ..., 1 :  0.9046526867619487  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd4/model_0017500.pth

    #max ..., 1 :  0.7642213642206758  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0025000.pth



    #max ..., 2 :  0.8884019011989823  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd0/model_0012500.pth
    #max ..., 2 :  0.8831241956233999  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1/model_0025000.pth
    #max ..., 2 :  0.821492647281454  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd2/model_0032500.pth
    #max ..., 2 :  0.8703693977582503  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd3/model_0060000.pth
    #max ..., 2 :  0.9201752948877534  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd4/model_0015000.pth

    #max ..., 2 :  0.7970881595874416  at  trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd1B/model_0025000.pth






