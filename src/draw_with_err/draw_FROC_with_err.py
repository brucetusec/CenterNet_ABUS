import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from numpy import trapz
from scipy.interpolate import interp2d, interp1d, interpnd

#data_from_file = r'npy/result_454_prop_newths_v3.npy'
data_from_file = r'performance.npy'
#folder_name = r'New_proposed_th'
folder_name = r'yolo_loss'

data = np.load(data_from_file)
# print(data)
#a = np.asarray(data[..., 2],data[..., 3])
np.savetxt("epoch60_sen_fp.csv", data, delimiter=",")
auc_list = []

font = {'family': 'Times New Roman',
        'size': 12}

plt.rc('font', **font)


def AUC(froc_x, froc_y, x_limit):
    global auc_list
    froc_x = np.array(froc_x)
    froc_y = np.array(froc_y)
    mask = np.logical_and(froc_y <= 1.0, froc_x <= x_limit)
    froc_x = froc_x[mask]
    froc_y = froc_y[mask]

    froc_x = np.array(froc_x) / froc_x[0]
    area = trapz(froc_y[::-1], x=froc_x[::-1])
    auc_list.append(area)
    return area


def draw_full(froc_x, froc_y, color, label, linestyle, x_limit, df):
    area = AUC(froc_x, froc_y, x_limit)
    plt.plot(froc_x, froc_y, color=color, label=label +
             ', Az = %.3f' % area, linestyle=linestyle)

    Y_inter = interp1d(froc_x, froc_y)
    x_mesh = np.arange(1, x_limit, 1)
    Y_interpolated = Y_inter(x_mesh)
    y_err = df['SEN_ERR'].to_numpy()
    plt.errorbar(x_mesh, Y_interpolated,
                 yerr=y_err[1:x_limit], color=color, fmt='.', ms=0, capsize=0)

#performnace_per_thre.append([score_hit_thre,total_pass,sensitivity, sum_FP/total_pass,sensitivity_IOU_1, sum_FP_IOU_1/total_pass,ChonHua_sensitivity, ChonHua_sum_FP/total_pass,ChonHua_sensitivity_1, ChonHua_sum_FP_1/total_pass,LUNA_sensitivity, LUNA_sum_FP/total_pass,sen0,sen1,sen2,big_sen0,big_sen1,big_sen2])


df_10MM = pd.read_csv(
    r'10mm_fix_th_stderr.csv', dtype=np.float, delimiter=',')
df_15MM = pd.read_csv(
    r'15mm_fix_th_stderr.csv', dtype=np.float, delimiter=',')
df_IOU25 = pd.read_csv(
    r'25percent_fix_th_stderr.csv', dtype=np.float, delimiter=',')
df_IOU10 = pd.read_csv(
    r'10percent_fix_th_stderr.csv', dtype=np.float, delimiter=',')


# draw_full(data[..., 3], data[..., 2], '#FF6D6C',
#           'IOU > 0.25 ', ':', 25, df_IOU25)
# draw_full(data[..., 5], data[..., 4], '#FF0000',
#           'IOU > 0.10 ', '-', 25, df_IOU10)
# draw_full(data[..., 9], data[..., 8], '#2E7FC1',
#           'D < 1.0cm ', ':', 25, df_10MM)
# draw_full(data[..., 7], data[..., 6], '#0000FF',
#           'D < 1.5cm ', '-', 25, df_15MM)
## draw_full(data[...,11],data[...,10],'#00B200', 'Distance < radius of GT', '-',25)

# axes = plt.gca()
# axes.set_xlim([0.125, 25])
# x_tick = np.arange(0, 25.01, 5)
# plt.xticks(x_tick)
# axes.set_ylim([0.5, 1.01])
# y_tick = np.arange(0.5, 1.01, 0.05)
# plt.yticks(y_tick)


# plt.legend(loc='lower right')
# plt.xlabel('False Positives Per Pass')
# plt.ylabel('Sensitivity')
# # plt.savefig(data_from_file.split('.',1)[0]+'_FP_to25.png', dpi=300)
# plt.show()


# performnace_per_thre.append([score_hit_thre,total_pass,
# sensitivity, sum_FP/total_pass,
# sensitivity_IOU_1, sum_FP_IOU_1/total_pass,
# ChonHua_sensitivity, ChonHua_sum_FP/total_pass,
# ChonHua_sensitivity_1 8, ChonHua_sum_FP_1/total_pass9,
# LUNA_sensitivity10, LUNA_sum_FP/total_pass11,
# sen0 12,sen1 13,sen2 14,
# big_sen0 15,big_sen1 16,big_sen2 17])

draw_full(data[..., 7], data[..., 6], '#2E7FC1',
          'IOU > 0.25 ', ':', 8, df_IOU25)
draw_full(data[..., 9], data[..., 8], '#0000FF',
          'IOU > 0.10 ', '-', 8, df_IOU10)
draw_full(data[..., 3], data[..., 2], '#FF6D6C', 'D < 1.0cm ', ':', 8, df_10MM)
draw_full(data[..., 5], data[..., 4], '#FF0000', 'D < 1.5cm ', '-', 8, df_15MM)
## draw_full(data[...,11],data[...,10],'#00B200', 'Distance < radius of GT', '-',8)

axes = plt.gca()
axes.set_xlim([0.125, 8])
x_tick = np.arange(0, 8.01, 1)
plt.xticks(x_tick)
axes.set_ylim([0.50, 1.01])
y_tick = np.arange(0.50, 1.01, 0.05)
plt.yticks(y_tick)


plt.legend(loc='lower right')
plt.xlabel('False Positives Per Pass')
plt.ylabel('Sensitivity')
# plt.savefig(data_from_file.split('.',1)[0]+'_FP_to8.png', dpi=300)

plt.show()
