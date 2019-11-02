import numpy as np
import abus_data

def main():
    all_data = abus_data.AbusNpyFormat(root, train=False, validation=False)
    print('Shape of all data:', np.shape(all_data.__getitem__(0)[0]))
    print('Data len:', all_data.__len__())
    print('Number of boxes of data[0]:', len(all_data.__getitem__(0)[1]))

    train_data = abus_data.AbusNpyFormat(root, train=True, validation=False)
    print('Shape of train data:', np.shape(train_data.__getitem__(0)[0]))
    print('Data len:', train_data.__len__())
    print('Number of boxes of data[0]:', len(train_data.__getitem__(0)[1]))

    val_data = abus_data.AbusNpyFormat(root, train=False, validation=True)
    print('Shape of val data:', np.shape(val_data.__getitem__(0)[0]))
    print('Data len:', val_data.__len__())
    print('Number of boxes of data[0]:', len(val_data.__getitem__(0)[1]))


if __name__ == '__main__':
    root = '../../data/sys_ucc/'
    main()