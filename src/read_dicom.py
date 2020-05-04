import os, argparse
import pydicom as dicom


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