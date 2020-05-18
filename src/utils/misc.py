import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom

def loadFileInformation(filename):
    information = {}
    ds = dicom.read_file(filename)

    information['NumberOfFrames'] = ds.NumberOfFrames if 'NumberOfFrames' in dir(ds) else 1
    information['PixelSpacing'] = ds.PixelSpacing if 'PixelSpacing' in dir(ds) else [1, 1]
    information['Rows'] = ds.Rows
    information['Columns'] = ds.Columns
    information['SliceThickness'] = ds.SliceThickness
    information['SpacingBetweenSlices'] = ds.SpacingBetweenSlices

    data = ds.pixel_array

    return [float(ds.SpacingBetweenSlices), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]


def AUC(froc_x, froc_y, x_limit, normalize=False):
    froc_x = np.array(froc_x)
    froc_y = np.array(froc_y)

    area = np.trapz(froc_y[::-1], x=froc_x[::-1], dx=0.001)

    if normalize:
        return area/np.max(froc_x[::-1])
    else:
        return area


def draw_full(froc_x, froc_y, color, label, linestyle, x_limit):
    area = AUC(froc_x, froc_y, x_limit)
    plt.plot(froc_x, froc_y, color=color, label=label +
             ', Az = %.3f' % area, linestyle=linestyle)


def build_threshold():
    thresholds = []
    
    tmp=0.005
    for i in range(0, 394):
        thresholds.append(tmp)
        tmp += 0.0025

    # for i in range(0, 75):
    #     thresholds.append(tmp)
    #     tmp += 0.01
     
    return thresholds