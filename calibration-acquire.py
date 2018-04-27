from imutils.video import WebcamVideoStream
import numpy as np
import cv2 as cv
import time
import imutils
import sys

if not (len(sys.argv) > 2):
    exit()
elif (sys.argv[1] == 'cam'):
    dest = 'cam'
    nimgs = 20
elif (sys.argv[1] == 'test'):
    dest = 'test'
    nimgs = 5
else:
    exit()

if (sys.argv[2] == 'ip'):
    src1 = 'rtsp://192.168.1.10:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    src2 = 'rtsp://192.168.1.11:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    folder = 'ipcam'
elif (sys.argv[2] == 'usb'):
    if not (len(sys.argv) > 4):
        exit()
    else:
        src1 = sys.argv[3]
        src2 = sys.argv[4]
        folder = 'usbcam'
else:
    exit()

counter = 1

cap1 = WebcamVideoStream(src=src1)
# cap1.stream.set(cv.CAP_PROP_FRAME_WIDTH, 640)
# cap1.stream.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

cap2 = WebcamVideoStream(src=src2)
# cap2.stream.set(cv.CAP_PROP_FRAME_WIDTH, 640)
# cap2.stream.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

cap1.start()
cap2.start()

time.sleep(4)

while(True):
    frame1 = cap1.read()
    frame2 = cap2.read()
    
    #cv.imshow('cam1', frame1)
    #cv.imshow('cam2', frame2)

    if int(time.time())%4 == 0:

        print 'frame %d' % counter
        print 'captured'

        # frame1 = cv.fastNlMeansDenoisingColored(frame1,None,10,10,7,21)
        # frame2 = cv.fastNlMeansDenoisingColored(frame2,None,10,10,7,21)
        # frame1 = cv.bitwise_not(cv.Canny(frame1,100,200))
        # frame2 = cv.bitwise_not(cv.Canny(frame2,100,200))
        # frame1 = cv.cvtColor(frame1, cv.COLOR_GRAY2BGR)
        # frame2 = cv.cvtColor(frame2, cv.COLOR_GRAY2BGR)

        cv.imwrite(('%s/%s1/img%d.jpg' % (folder,dest, counter)),frame1)
        cv.imwrite(('%s/%s2/img%d.jpg' % (folder,dest, counter)),frame2)
        print 'saved'
        counter = counter + 1
        time.sleep(1)

    cv.imshow('cam1', frame1)

    if (counter > nimgs) or (cv.waitKey(1) & 0xFF == ord('q')):
        break


cap1.stop()
cap2.stop()
cv.destroyAllWindows()
