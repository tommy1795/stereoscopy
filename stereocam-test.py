from imutils.video import WebcamVideoStream
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
        src1 = sys.argv[2]
        src2 = sys.argv[3]
    else:
        exit()
except:
    exit()

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
print (w, h)

rear_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')#cascades/haarcascade_car_rear.xml')

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
time.sleep(4)

while(True):
    frame1 = cap1.read()
    frame2 = cap2.read()
    
    fixed1 = cv.remap(frame1, cam1mapx, cam1mapy, cv.INTER_LINEAR)
    fixed2 = cv.remap(frame2, cam2mapx, cam2mapy, cv.INTER_LINEAR)
    
    #dst1 = cv.undistort(frame1, cam1mtx, cam1dis, None, cam1proj)
    #dst2 = cv.undistort(frame2, cam2mtx, cam2dis, None, cam2proj)

    gray1 = cv.cvtColor(fixed1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(fixed2, cv.COLOR_BGR2GRAY)
    
    stereo = cv.StereoBM_create()
    stereo.setMinDisparity(4)
    stereo.setNumDisparities(128)
    stereo.setBlockSize(25)
    stereo.setSpeckleRange(16)
    stereo.setSpeckleWindowSize(45)


    disparity = stereo.compute(gray1,gray2).astype(np.float32)

    img3d = cv.reprojectImageTo3D(disparity, disp2depth)
    # img3d *= .024

    cellsize = 20
    font = cv.FONT_HERSHEY_SIMPLEX
    
    #for i in range(0, w/cellsize):
    #    for j in range(0, h/cellsize):
    #        avgdist = 1./np.mean(disparity[20*j:20*(j+1),20*i:20*(i+1)])*1400
    #        text = '%.1f' % avgdist
    #        cv.putText(fixed2,text,(20*i,20*(j+1)), font, .2,(255,255,255),1,cv.LINE_AA)
    
    #print(img3d[370,550,2])
    img3d = cv.reprojectImageTo3D(disparity, disp2depth).astype(np.float32)
    # print(img3d[235:245,315:325,2])
    # print(1./np.mean(disparity[220:260,300:340])*1.)

    cars = rear_cascade.detectMultiScale(gray1, 1.2, 5)
    
    for (x,y,w,h) in cars:
        cv.rectangle(fixed1,(x,y),(x+w,y+h),(255,0,0),2)
        # calculate coordinates for averaging
        k = .7 #coefficient
        x1 = int(x+(1-k)/2*w)
        x2 = int(x+(1+k)/2*w)
        y1 = int(y+(1-k)/2*h)
        y2 = int(y+(1+k)/2*h)
        
        avgdist1 = cam_f * car_w / w #1./np.amax(disparity[y1:y2,x1:x2])*3600
        avgdist2 = np.mean(img3d[y1:y2,x1:x2,2])*9.35+0.7242
        avgdist1 = avgdist1 * 0.6794 + 0.16826
        text1 = '%.1f' % avgdist1
        text2 = '%.1f' % avgdist2
        cv.putText(fixed1,text1,(x1,y1), font, 4,(0,255,0),4,cv.LINE_AA)
        cv.putText(fixed1,text2,(x2,y2), font, 4,(255,0,255),8,cv.LINE_AA)

    cv.imshow('depth', disparity/2048.)
    

    # dst = cv.undistort(frame1, cam1mtx, cam1dis, None, newcam1mtx)

    # dst = imutils.rotate(dst, angle=180)

    # gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    
    #cars = rear_cascade.detectMultiScale(gray1, 1.3, 5)
    #for (x,y,w,h) in cars:
    #    cv.rectangle(dst1,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv.imshow('fixed1', fixed1)
    cv.imshow('fixed2', fixed2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# print(img3d.shape)
cap1.stop()
cap2.stop()

cv.destroyAllWindows()
