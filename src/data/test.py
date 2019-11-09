from abus_data import AbusNpyFormat

def main():
    all_data = AbusNpyFormat(root, train=False, validation=False)
    data, label = all_data.__getitem__(0)
    print('Shape of data:', data.size())
    print('Data len:', all_data.__len__())
    print('Number of boxes in data[0]:', len(label))
    print('Volumetric tensor:', data)
    print('Label:', label)

if __name__ == '__main__':
    root = '../../data/sys_ucc/'
    main()