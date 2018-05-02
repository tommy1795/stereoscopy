from threading import Thread
import numpy as np
import cv2 as cv

class FCWwatchdog:
    def __init__(self):
        # initialize calculated velocity and required acceleration
        self.normal = np.zeros((512,512))
        self.caution = cv.imread('misc/caution.jpg', 1)
        self.warning = cv.imread('misc/warning.jpg', 1)
        self.image = self.normal
        self.stopped = False
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                cv.destroyAllWindows()
                return
            cv.imshow('status', self.image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True        
                break
    def recalc(self, velocity=0.0, z=50.0):
        if (velocity < 0):
            deceleration = velocity * velocity / z / 2.0
            if (deceleration > 0.5):
                self.image = self.warning
            else:
                self.image = self.caution
        else:
            self.image = self.normal
    def stop(self):
        self.stopped = True

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
