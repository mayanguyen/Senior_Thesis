### Van Mai Nguyen Thi
### Senior Project
### source: http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import numpy as np
import cv2
import glob

def chessCornersForCamera(camId, version):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:8].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    corners2 = np.zeros((5*8,2))
    
    images = glob.glob('./data/test'+str(version)+'/chessP*'+str(camId)+'.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray,(8,5),None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            imgpoints.append(corners)
            corners2 = np.asarray(corners)
            
            # Draw and save the corners
            cv2.drawChessboardCorners(img, (8,5), corners,ret)
            cv2.imwrite('./data/test'+str(version)+'/chessCorners'+str(camId)+'.jpg', img)
            #cv2.waitKey(500)

    np.ravel(corners2)
    corners2 = corners2.reshape((-1,2))

    return corners2

def chessCorners(version):
    corners1 = chessCornersForCamera('1', version)
    corners2 = chessCornersForCamera('2', version)
    return corners1, corners2



