### Van Mai Nguyen Thi
### Senior Project
### source: http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import numpy as np
import cv2
import glob

def camCalib():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:8].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    corners2 = np.zeros((5*8,2))
    
    images = glob.glob('./data/chessP*.jpg')
    #print(len(images))
    
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
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,5), corners,ret)
            #cv2.imshow('img',img)
            cv2.imwrite('./data/chessCorners.jpg', img)
            cv2.waitKey(500)

    '''k = cv2.waitKey(0)
    if k == 27:
    cv2.destroyAllWindows()'''
    
    np.ravel(corners2)
    corners2 = corners2.reshape((-1,2))

    return corners2


