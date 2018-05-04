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

''' probably not needed
chsbd_w = 0.2355
chsbd_p = 578
chsbd_d = 0.88
cam_f = chsbd_p * chsbd_d / chsbd_w
car_w = 1.8
'''

# coefficients for raw linear distance calculation
m1 = 2674.47615612931
b1 = 0.626967681334889
m2 = -2.40135001577826
b2 = 17.8185160919775
c2 = -2.07837059967143


# Kalman filter initialization
dist = 50.0
dist_pri = 50.0
dist_post = 50.0
P_pri = 1.0
P_post = 1.0
Q = 10e-3
def var_stereo(dist):
    #var = .00625715 * dist * dist + 0.2544023 * dist
    var = 0.024303279394464 * dist + -0.114193003893593
    if (var < 0.052114226006177):
        var = 0.052114226006177
    return var
var_prop = 2.24022762915913
A = 1.0
C = np.array([[1],[1]])
velo = 0.0

normal = np.zeros((512,512))
caution = cv.imread('misc/caution.jpg', 1)
warning = cv.imread('misc/warning.jpg', 1)
image = normal



''' not needed if my fcn works
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

# print (w, h)

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

#wd = fcns.FCWwatchdog().start()

start_time = time.time()

while(True):
    frame1 = cap1.read()
    frame2 = cap2.read()
    
    fixed1 = cv.remap(frame1, cam1mapx, cam1mapy, cv.INTER_LINEAR)
    fixed2 = cv.remap(frame2, cam2mapx, cam2mapy, cv.INTER_LINEAR)
    gray1 = cv.cvtColor(fixed1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(fixed2, cv.COLOR_BGR2GRAY)
    #except:
        #gray1 = cv.imread('misc/blank.jpg', 1)
        #gray2 = cv.imread('misc/blank.jpg', 1)
        #gray1 = cv.cvtColor(gray1, cv.COLOR_BGR2GRAY)
        #gray2 = cv.cvtColor(gray2, cv.COLOR_BGR2GRAY)
        #pass
    
    stereo = cv.StereoBM_create()
    stereo.setMinDisparity(2) # 4
    stereo.setNumDisparities(64) # 128
    stereo.setBlockSize(21)
    stereo.setSpeckleRange(16) # 16
    stereo.setSpeckleWindowSize(45) # 45


    disparity = stereo.compute(gray1,gray2).astype(np.float32)

    img3d = cv.reprojectImageTo3D(disparity, disp2depth)

    cellsize = 20
    font = cv.FONT_HERSHEY_SIMPLEX
    
    img3d = cv.reprojectImageTo3D(disparity, disp2depth).astype(np.float32)

    cars = rear_cascade.detectMultiScale(gray1, 1.2, 5)
    
    dist1 = 100.
    dist2 = 100.    

    for (x,y,w,h) in cars:
        cv.rectangle(fixed1,(x,y),(x+w,y+h),(255,0,0),2)
        # calculate coordinates for averaging
        k = .7 #coefficient
        
        x1, x2, y1, y2 = fcns.img_subregion(x, y, w, h, k)
        # calculate the distance using two methods
        avgdist1 = m1 / np.float64(w) + b1
        avgdist2 = np.amin(img3d[y1:y2,x1:x2,2])
        avgdist2 = avgdist2 * avgdist2 * m2 + avgdist2 * b2 + c2

        # dim case, closest found using proportions is probably a car
        if (avgdist1 < dist1):
            dist1 = avgdist1
            dist2 = avgdist2

        #distances for debugging
        #text1 = '%d' % w
        #text2 = '%.4f' % np.amin(img3d[y1:y2,x1:x2,2])
        #cv.putText(fixed1,text1,(x1,y1), font, 2,(0,255,0),4,cv.LINE_AA)
        #cv.putText(fixed1,text2,(x2,y2), font, 2,(255,0,255),8,cv.LINE_AA)
        

    cv.putText(fixed1,'%.1f' % dist1,(50,800), font, 2,(0,255,0),8,cv.LINE_AA)
    cv.putText(fixed1,'%.1f' % dist2,(50,900), font, 2,(255,0,255),8,cv.LINE_AA)

    # Kalman filter prediction
    dist_pri = A * dist_post
    P_pri = P_post + Q
    velo = dist_post

    # Kalman filter update
    R = np.array([[var_prop, 10e-5], [10e-5, var_stereo(dist_pri)]])
    S = np.dot(P_pri*C,C.transpose()) + R
    K = np.dot(P_pri*C.transpose(),np.linalg.inv(S))
    e = np.array([[dist1],[dist2]]) - dist_pri*C

    dist_post = dist_pri + np.dot(K, e)
    P_post = P_pri - np.dot(np.dot(K, S), K.transpose())
    
    # Calculate the velocity and update the watchdog
    velo = (dist_post - velo)/(time.time() - start_time)
    start_time = time.time()
    #wd.recalc(velo, dist_post)
    if (velo<-5.0/3.6 and dist_post > 5.5):
        acc = velo*velo/(dist_post-5.0)/2.0 # added one car margin
        if (acc > 1): #TODO changed acceleration to 4 m/s2
            image = warning
        else:
            image = caution
    else:
        image = normal
        acc = 0.0

    cv.putText(fixed1,'%.1f' % dist_post,(50,700), font, 2,(255,0,0),8,cv.LINE_AA)
    cv.putText(fixed1,'%.1f' % velo,(50,600), font, 2,(255,255,0),8,cv.LINE_AA)
    cv.putText(fixed1,'%.1f' % acc,(50,500), font, 2,(0,255,255),8,cv.LINE_AA)

    cv.imshow('status', image)    
    cv.imshow('depth', disparity/1024.)
    
    cv.imshow('fixed1', fixed1)
    #cv.imshow('fixed2', fixed2)

    if cv.waitKey(1) & 0xFF == ord('q'):        
        break

# print(img3d.shape)
cap1.stop()
cap2.stop()
#wd.stop()

cv.destroyAllWindows()
