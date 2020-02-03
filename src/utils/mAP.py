import numpy as np
import matplotlib.pyplot as plt

data_from_file = 'testing_result/result.npy'

data = np.load(data_from_file)
auc_list = []

font = {'family': 'Times New Roman',
        'size': 12}

plt.rc('font', **font)

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


draw_full(data[..., 2], data[..., 3], '#FF6D6C', 'IOU > 0.25 ', ':', 1)
draw_full(data[..., 5], data[..., 6], '#FF0000', 'IOU > 0.10 ', '-', 1)

axes = plt.gca()
axes.set_xlim([0, 1])
x_tick = np.arange(0, 1, 0.125)
plt.xticks(x_tick)
axes.set_ylim([0, 1])
y_tick = np.arange(0, 1, 0.125)
plt.yticks(y_tick)


plt.legend(loc='lower right')
plt.ylabel('Precision')
plt.xlabel('Sensitivity')
plt.show()