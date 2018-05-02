from imutils.video import WebcamVideoStream
import stereocamfcn as fcns
import numpy as np
import cv2 as cv
import time
import imutils
import sys

try:
    if (sys.argv[1] == 'ip'):
        src1 = 'rtsp://192.168.1.10:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        src2 = 'rtsp://192.168.1.11:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        folder = 'ipcam'
    elif (sys.argv[1] == 'usb'):
        folder = 'usbcam'
        src1 = int(sys.argv[2])
        src2 = int(sys.argv[3])
    else:
        exit()
except:
    exit()

chsbd_w = 0.2355
chsbd_p = 578
chsbd_d = 0.88
cam_f = chsbd_p * chsbd_d / chsbd_w
car_w = 1.8

m1 = 3333.4004557751
b1 = -2.34340798146414
m2 = 10.9119480056585
b2 = 3.37950280275768

pdist1 = 50.
pdist2 = 50.
velo1 = 0.
velo2 = 0.


'''
calibration = np.load('%s/calib-stereo.npz' % folder , allow_pickle=False)
(w, h) = tuple(calibration["imageSize"])
cam1mtx = calibration["mtx1"]
cam1dis = calibration["dist1"]
cam1mtxn = calibration["newmtx1"]
cam1proj = calibration["proj1"]
cam1mapx = calibration["mapx1"]
cam1mapy = calibration["mapy1"]
leftROI = tuple(calibration["roi1"])
cam2mtx = calibration["mtx2"]
cam2dis = calibration["dist2"]
cam2mtxn = calibration["newmtx2"]
cam2proj = calibration["proj2"]
cam2mapx = calibration["mapx2"]
cam2mapy = calibration["mapy2"]
rightROI = tuple(calibration["roi2"])
disp2depth = calibration["disp2depth"]
'''
cam1mapx, cam1mapy, cam2mapx, cam2mapy, disp2depth = fcns.readcalibration('%s/calib-stereo.npz' % folder)

print (w, h)

rear_cascade = cv.CascadeClassifier('cascades/haarcascade_car_rear.xml')#frontalface_alt.xml')

cap1 = WebcamVideoStream(src=src1)
# cap1.stream.set(cv.CAP_PROP_FRAME_WIDTH, w)
# cap1.stream.set(cv.CAP_PROP_FRAME_HEIGHT, h)
# cap1.stream.set(cv.CAP_PROP_FPS, 10)

cap2 = WebcamVideoStream(src=src2)
# cap2.stream.set(cv.CAP_PROP_FRAME_WIDTH, w)
# cap2.stream.set(cv.CAP_PROP_FRAME_HEIGHT, h)
# cap2.stream.set(cv.CAP_PROP_FPS, 10)

cap1.start()
cap2.start()
time.sleep(2)

wd = fcns.FCWwatchdog().start()

start_time = time.time()

while(True):
    frame1 = cap1.read()
    frame2 = cap2.read()
    
    fixed1 = cv.remap(frame1, cam1mapx, cam1mapy, cv.INTER_LINEAR)
    fixed2 = cv.remap(frame2, cam2mapx, cam2mapy, cv.INTER_LINEAR)

    gray1 = cv.cvtColor(fixed1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(fixed2, cv.COLOR_BGR2GRAY)
    
    stereo = cv.StereoBM_create()
    stereo.setMinDisparity(4) # 4
    stereo.setNumDisparities(32) # 128
    stereo.setBlockSize(21)
    stereo.setSpeckleRange(16) # 16
    stereo.setSpeckleWindowSize(45) # 45


    disparity = stereo.compute(gray1,gray2).astype(np.float32)

    img3d = cv.reprojectImageTo3D(disparity, disp2depth)

    cellsize = 20
    font = cv.FONT_HERSHEY_SIMPLEX
    
    img3d = cv.reprojectImageTo3D(disparity, disp2depth).astype(np.float32)

    cars = rear_cascade.detectMultiScale(gray1, 1.2, 5)
    
    dist1 = 50.
    dist2 = 50.    

    for (x,y,w,h) in cars:
        cv.rectangle(fixed1,(x,y),(x+w,y+h),(255,0,0),2)
        # calculate coordinates for averaging
        k = .7 #coefficient
        '''        
        x1 = int(x+(1-k)/2*w)
        x2 = int(x+(1+k)/2*w)
        y1 = int(y+(1-k)/2*h)
        y2 = int(y+(1+k)/2*h)
        '''
        x1, x2, y1, y2 = fcns.img_subregion(x, y, w, h, k)
        # calculate the distance using two methods
        avgdist1 = m1 / np.float64(w) + b1
        avgdist2 = np.median(img3d[y1:y2,x1:x2,2]) * m2 + b2

        # dim case, closest found using proportions is probably a car
        if (avgdist1 < dist1):
            dist1 = avgdist1
            dist2 = avgdist2

        text1 = '%d' % w
        text2 = '%.4f' % avgdist2
        cv.putText(fixed1,text1,(x1,y1), font, 2,(0,255,0),4,cv.LINE_AA)
        cv.putText(fixed1,text2,(x2,y2), font, 2,(255,0,255),8,cv.LINE_AA)
    # TODO
    velo1 = [] 

    cv.imshow('depth', disparity/512.)
    
    cv.imshow('fixed1', fixed1)
    #cv.imshow('fixed2', fixed2)

    if cv.waitKey(1) & 0xFF == ord('q'):        
        break

# print(img3d.shape)
cap1.stop()
cap2.stop()
wd.stop()

cv.destroyAllWindows()
