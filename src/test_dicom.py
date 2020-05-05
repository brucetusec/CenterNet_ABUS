import os, argparse
import pydicom as dicom
from scipy import ndimage


def loadFileInformation(filename):
    information = {}
    ds = dicom.read_file(filename)

    information['NumberOfFrames'] = ds.NumberOfFrames if 'NumberOfFrames' in dir(
        ds) else 1
    information['PixelSpacing'] = ds.PixelSpacing if 'PixelSpacing' in dir(ds) else [
        1, 1]
    information['Rows'] = ds.Rows
    information['Columns'] = ds.Columns
    information['SliceThickness'] = ds.SliceThickness
    information['SpacingBetweenSlices'] = ds.SpacingBetweenSlices

    data = ds.pixel_array

    return [data, float(ds.SpacingBetweenSlices), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]


def main(args): 
    SCALE = 1
    img_vol, z_PixelSpacing, y_PixelSpacing, x_PixelSpacing = loadFileInformation("./dicom/new_CASE/SR_Cai^Shunping_965_201902141326/1.3.6.1.4.1.47779.1.002.dcm")
    print([z_PixelSpacing*SCALE, y_PixelSpacing*SCALE, x_PixelSpacing*SCALE])
    print(img_vol.shape)
    img_vol = ndimage.interpolation.zoom(img_vol, [z_PixelSpacing*SCALE, y_PixelSpacing*SCALE, x_PixelSpacing*SCALE], mode='nearest')
    print(img_vol.shape)
    return 


def _parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--scale', '-s', type=int, default=1,
    #     help='How much were x,z downsampled?'
    # )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args)