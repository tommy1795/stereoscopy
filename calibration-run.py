import numpy as np
import cv2
import time
import sys

if not (len(sys.argv) == 4):
    exit()
if (sys.argv[1] == 'ip'):
    folder = 'ipcam'
elif (sys.argv[1] == 'usb'):
    folder = 'usbcam'
else:
    exit()

#grid parameters
m = int(sys.argv[2])
n = int(sys.argv[3])

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((n*m,3), np.float32)
objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2) #np.mgrid[0:6*24:7j,0:5*24:6j].T.reshape(-1,2)
squaresize = 0.0393
objp *= squaresize

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.
imgpoints2 = [] # 2d points in image plane.

imgno = 1
nimgs = 20

while(imgno<=nimgs):
    path1 = '%s/cal1/img%d.jpg' % (folder, imgno)
    path2 = '%s/cal2/img%d.jpg' % (folder, imgno)
    img1 = cv2.imread(path1, 1)
    img2 = cv2.imread(path2, 1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, (m,n),None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, (m,n),None)
    
    # If found, add object points, image points (after refining them)
    if ret1 == True and ret2 == True:
        objpoints.append(objp)
        
        corners21 = cv2.cornerSubPix(gray1,corners1,(11,11),(-1,-1),criteria)
        imgpoints1.append(corners1)
        corners22 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
        imgpoints2.append(corners2)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img1, (m,n), corners21,ret1)
        cv2.drawChessboardCorners(img2, (m,n), corners22,ret2)
        cv2.imshow('img1', img1)
        cv2.waitKey(100)
        cv2.imshow('img2', img2)
        cv2.waitKey(100)
        # cv2.imshow('img1',img1)
        # cv2.imshow('img2',img2)
        # time.sleep(50)
        # cv2.waitKey(500)
    
    # Go to the next file
    imgno += 1

# cv2.destroyAllWindows()
print(gray1.shape[::-1])

# Calculate the parameters of both cameras with zero tangential dist, then reduce the vector size
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1],None,None)#,None,None,cv2.CALIB_ZERO_TANGENT_DIST)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1],None,None)#,None,None,cv2.CALIB_ZERO_TANGENT_DIST)
# dist1 = dist1[:,0:4]
# dist2 = dist2[:,0:4]

# Get new optimal matrices
newmtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, gray1.shape[::-1], 1)
newmtx2, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, dist2, gray1.shape[::-1], 1)

# Calculate the rotatnion matrix and translation vector between the two cameras
(_, _, _, _, _, rotmtx, transvec, _, fundmtx) = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, 
    mtx1, dist1,
    mtx2, dist2,
    gray1.shape[::-1],
    None, None,
    None, None,
    cv2.CALIB_FIX_INTRINSIC,  criteria)

# Calculate the rectification parameters
(rect1, rect2, proj1, proj2, disp2depth, roi1, roi2) = cv2.stereoRectify(
    mtx1, dist1, mtx2, dist2,
    gray1.shape[::-1], rotmtx, transvec,
    None, None, None, None, None,
    0, 0.5)

# ip1 = np.asarray(imgpoints1)
# ip1 = np.reshape(ip1, (ip1.shape[0]*ip1.shape[1],ip1.shape[3]))

# ip2 = np.asarray(imgpoints2)
# ip2 = np.reshape(ip2, (ip2.shape[0]*ip2.shape[1],ip2.shape[3]))

# (_, h1, h2) = cv2.stereoRectifyUncalibrated(ip1, ip2, fundmtx, gray1.shape[::-1])
# rec1 = np.linalg.inv(mtx1)*h1*mtx1
# rec2 = np.linalg.inv(mtx2)*h2*mtx2

# Calculate the rectification matrices
mapx1, mapy1 = cv2.initUndistortRectifyMap(mtx1, dist1, rect1, proj1, gray1.shape[::-1], cv2.CV_32FC1)
mapx2, mapy2 = cv2.initUndistortRectifyMap(mtx2, dist2, rect2, proj2, gray1.shape[::-1], cv2.CV_32FC1)

# Save the calibration parameters
np.savez_compressed(
    '%s/calib-stereo.npz' % folder, imageSize=gray1.shape[::-1],
    mtx1=mtx1, dist1=dist1, newmtx1=newmtx1, proj1=proj1, mapx1=mapx1, mapy1=mapy1, roi1=roi1,
    mtx2=mtx2, dist2=dist2, newmtx2=newmtx2, proj2=proj2, mapx2=mapx2, mapy2=mapy2, roi2=roi2,
    disp2depth=disp2depth)

print(mtx1)
print(dist1)
print(mtx2)
print(dist2)
print(rotmtx)
print(transvec)
print(fundmtx)
