import numpy as np

def readcalibration(calibrationpath):
    calibration = np.load(calibrationpath , allow_pickle=False)
    cam1mapx = calibration["mapx1"]
    cam1mapy = calibration["mapy1"]
    cam2mapx = calibration["mapx2"]
    cam2mapy = calibration["mapy2"]
    disp2depth = calibration["disp2depth"]
    return (cam1mapx, cam1mapy, cam2mapx, cam2mapy, disp2depth)

def img_subregion(x, y, w, h, k):
    x1 = int(x+(1-k)/2*w)
    x2 = int(x+(1+k)/2*w)
    y1 = int(y+(1-k)/2*h)
    y2 = int(y+(1+k)/2*h)
    return (x1, x2, y1, y2)

def distance_reprod(w, m, b):
    d = m / w + b
    return d

def distance_stereo(img3d_mtx, x1, x2, y1, y2, m, b):
    d = np.amin(img3d_mtx[y1:y2,x1:x2,2]) * m + b
    return d
