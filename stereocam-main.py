from imutils.video import WebcamVideoStream
import stereocamfcn as fcns
import numpy as np
import cv2 as cv
import time
import imutils
import sys

timestr = time.strftime("%Y%m%d-%H%M%S")
fname = 'logs/' + timestr + '.txt'
logfile = open(fname, 'w')
logfile.write('time\td_stereo\td_prop\td_filter\tvelo\tacc\tstatus\n')

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

# read camera calibration
cam1mapx, cam1mapy, cam2mapx, cam2mapy, disp2depth = fcns.readcalibration('%s/calib-stereo.npz' % folder)

# coefficients for raw linear distance calculation
if (sys.argv[1] == 'ip'):
    m1 = np.float64(2674.47615612931)
    b1 = np.float64(0.626967681334889)
    a2 = np.float64(0.0) #-2.40135001577826)
    m2 = np.float64(9.46510215216222) #17.8185160919775)
    b2 = np.float64(3.19819909774238) #-2.07837059967143)
else:
    m1 = np.float64(2167.155186688)
    b1 = np.float64(-5.7464734246)
    a2 = np.float64(0.0) #37.4122794202)
    m2 = np.float64(65.1769726763979) #32.1023862723)
    b2 = np.float64(-14.4336601527706) #-7.5484309991)

# read classifier parameters
rear_cascade = cv.CascadeClassifier('cascades/haarcascade_car_rear.xml')

# create and setup stereo matcher object
stereo = cv.StereoBM_create()
stereo.setMinDisparity(2) # 4
stereo.setNumDisparities(64) # 128
stereo.setBlockSize(21)
stereo.setSpeckleRange(16) # 16
stereo.setSpeckleWindowSize(45) # 45

# Kalman filter initialization
dist = np.float64(50.0)
dist_pri = np.float64(50.0)
dist_post = np.float64(50.0)
P_pri = np.float64(1.0)
P_post = np.float64(1.0)
Q = np.float64(10e-3)
def var_stereo(dist):
    #var = .00625715 * dist * dist + 0.2544023 * dist
    var = np.float64(0.024303279394464) * dist + np.float64(-0.114193003893593)
    if (var < 0.052114226006177):
        var = np.float64(0.052114226006177)
    return var
var_prop = np.float64(0.355514373263418) #2.24022762915913) # checking median
A = np.float64(1.0)
C = np.array([[1],[1]])
velo = np.float64(0.0)

dist1 = np.float64(100.0)
dist2 = np.float64(100.0)

# read status images
normal = np.zeros((512,512))
caution = cv.imread('misc/caution.jpg', 1)
warning = cv.imread('misc/warning.jpg', 1)
image = normal
st_int = 0

# setup threshold cnstants
dist_thr = 5.5
velo_thr = -5.0/3.6
acc_thr = 1.0
len_car = 5.0

# setup font for data display
font = cv.FONT_HERSHEY_SIMPLEX

# create video stream threads
cap1 = WebcamVideoStream(src=src1)
cap2 = WebcamVideoStream(src=src2)

# define usbcam stream properties
try:
    if (sys.argv[2] == 'usb'):
        cap1.stream.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap1.stream.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap2.stream.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap2.stream.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
except:
    pass

# start the streams
cap1.start()
cap2.start()
time.sleep(2)

# read the start time for the loop timing
start_time = time.time()

while(True):
    # read the images from L & R cameras
    frame1 = cap1.read()
    frame2 = cap2.read()
    
    # undistort and convert to grayscale
    fixed1 = cv.remap(frame1, cam1mapx, cam1mapy, cv.INTER_LINEAR)
    fixed2 = cv.remap(frame2, cam2mapx, cam2mapy, cv.INTER_LINEAR)
    gray1 = cv.cvtColor(fixed1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(fixed2, cv.COLOR_BGR2GRAY)

    # match the images, calculate disparity & 3d reprojection
    disparity = stereo.compute(gray1,gray2).astype(np.float32)
    img3d = cv.reprojectImageTo3D(disparity, disp2depth).astype(np.float32)
    
    # resize the image for object detection & apply the classifier
    gray_small = gray1[::2,::2]
    cars = rear_cascade.detectMultiScale(gray_small, 1.2, 5)

    # variables to store the minimum distances from the image processing
    temp1 = np.float64(100.0)
    temp2 = np.float64(100.0)
    no_car_found = np.bool_(True)

    for (x,y,w,h) in cars:
        # detection is on 2x smaller picture, revert coordinates x2
        x *= 2
        y *= 2
        w *= 2
        h *= 2
        # calculate distance by proportion 
        avgdist1 = m1 / np.float64(w) + b1

        # calculate distance with disparity
        k = .65 #coefficient        
        x1, x2, y1, y2 = fcns.img_subregion(x, y, w, h, k)
        avgdist2 = np.amin(img3d[y1:y2,x1:x2,2])
        avgdist2 = avgdist2 * avgdist2 * a2 + avgdist2 * m2 + b2

        # check if the data is consistent
        if (np.abs((avgdist1-avgdist2)/avgdist1)>0.75 and np.abs((avgdist1-avgdist2)/avgdist2)>0.75):
            continue
        elif (avgdist1 < 0.0 or avgdist2 < 0.0):
            continue
        else:
            # draw the rectangle for processed object and stereoscopy subregion
            cv.rectangle(fixed1,(x,y),(x+w,y+h),(255,0,0),2)
            cv.rectangle(fixed1,(x1,y1),(x2,y2),(0,255,0),2)
        
            # get the smallest
            if (avgdist2 < temp2):
                temp1 = avgdist1
                temp2 = avgdist2
                no_car_found = False

            '''#uncomment for calibration
            text1 = '%d' % w
            text2 = '%.4f' % np.amin(img3d[y1:y2,x1:x2,2])
            cv.putText(fixed1,text1,(x1,y1), font, 2,(0,255,0),4,cv.LINE_AA)
            cv.putText(fixed1,text2,(x2,y2), font, 2,(255,0,255),4,cv.LINE_AA)
            '''
    
    # interpret the measured values
    if (no_car_found): #if there was no car recognition
        if (dist1 < 35.0 and dist2 < 35.0): # if the full ahead space is not free
            dist1 += 0.2 # assume that there is nothing ahead and gradually increase the clear distance
            dist2 += 0.2
    else: #if there was car found, update the measurements for the Kalman
        dist1 = temp1
        dist2 = temp2
    
    # show the measurements on screen
    cv.putText(fixed1,'%.1f m pr' % dist1,(50,800), font, 2,(0,255,0),8,cv.LINE_AA)
    cv.putText(fixed1,'%.1f m st' % dist2,(50,900), font, 2,(255,0,255),8,cv.LINE_AA)

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
    
    # Calculate the velocity and update the status
    velo = (dist_post - velo)/(time.time() - start_time)
    start_time = time.time()
    
    if (velo<velo_thr and dist_post > dist_thr):
        acc = velo*velo/(dist_post-len_car)/2.0
        if (acc > acc_thr):
            image = warning
            st_int = 2
        else:
            image = caution
            st_int = 1
    else:
        image = normal
        st_int = 0
        acc = 0.0

    logfile.write('%f\t%f\t%f\t%f\t%f\t%f\t%d\n' % (time.time(), dist2, dist1, dist_post, velo, acc, st_int))

    # show results on screen
    cv.putText(fixed1,'%.1f m' % dist_post,(50,700), font, 2,(255,0,0),8,cv.LINE_AA)
    cv.putText(fixed1,'%.1f m/s' % velo,(50,600), font, 2,(255,255,0),8,cv.LINE_AA)
    cv.putText(fixed1,'%.1f m/s2' % acc,(50,500), font, 2,(0,255,255),8,cv.LINE_AA)

    cv.imshow('status', image)    
    cv.imshow('depth', disparity[::2,::2]/1024.)
    cv.imshow('fixed1', fixed1)

    if cv.waitKey(1) & 0xFF == ord('q'):        
        break

# cleanup
cap1.stop()
cap2.stop()
cv.destroyAllWindows()
logfile.close()
