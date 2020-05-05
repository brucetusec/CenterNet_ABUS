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