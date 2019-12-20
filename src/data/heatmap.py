import numpy as np

def _embed_matrix(A, B, start_point):
    A[start_point[0]:start_point[0]+B.shape[0], start_point[1]:start_point[1]+B.shape[1], start_point[2]:start_point[2]+B.shape[2]] += B
    return A

# Create a 2D numpy array simulating 2D gaussian dist.
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian3D(shape, sigma=1):
    r3, r2, r1 = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-r3:r3+1,-r2:r2+1,-r1:r1+1]

    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gen_3d_heatmap(size, gt_boxes, scale=1):
    size = [w//scale for w in size]
    hm = np.zeros(size, dtype=np.float32)

    for bbox in gt_boxes:
        half_width = (int((bbox['z_range']/2)//scale), int((bbox['y_range']/2)//scale), int((bbox['x_range']/2)//scale))
        half_width = [w if w > 0 else 1 for w in half_width]
        gauss_3d = gaussian3D(half_width)
        layer = np.zeros(size, dtype=np.float32)
        start_point = (int(bbox['z_center']//scale - half_width[0]//2), int(bbox['y_center']//scale - half_width[1]//2), int(bbox['x_center']//scale - half_width[2]//2))
        end_point = (int(bbox['z_center']//scale + half_width[0]//2), int(bbox['y_center']//scale + half_width[1]//2), int(bbox['x_center']//scale + half_width[2]//2))
        print('start:', start_point, 'end:', end_point, 'center:', (bbox['z_center']//scale, bbox['y_center']//scale, bbox['x_center']//scale), 'half-w:', half_width)
        layer = _embed_matrix(layer, gauss_3d, start_point)
        hm = np.maximum(hm, layer)

    return hm