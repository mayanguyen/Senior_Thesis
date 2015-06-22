'''
    This file was downloaded from moodle2.bard.edu, Computational Image course.
    By Keith O'Hara?
    
    Modified by Van Mai Nguyen Thi
    November 2014
    
    relevant opencv documentation:
    http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
    '''

import numpy as np
import cv2

MAX_FRAMES = 50
NULL_STRING = 'null'

def takePic(fname, scale):
    if scale <= 0:
        scale = 1

    start = False
    
    cap1 = cv2.VideoCapture(0)
    width = cap1.get(3)        # get width
    height = cap1.get(4)       # get height
    width, height = round(scale*width), round(scale*height)

    ret, frame_temp = cap1.read()
    d = 0
    frame = np.zeros( (height, width, 3), np.uint8) # empty black image
    frame2 = np.zeros( (height, width, 3), np.uint8) # empty black image

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    
    # Wait for ' ' (space) to start taking pictures
    while (start == False):
        ret, frame_temp = cap1.read()
        cv2.resize(frame_temp, (0,0), frame, scale, scale)
        
        cv2.putText(frame, 'Press SPACE to start taking pictures of '+fname,
                    (int(width/10.0),int(height*9.0/10)),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255))
        cv2.imshow('camera',frame)
                    
        k = cv2.waitKey(10) & 0xFF
        if k == ord(' '):   # space
            d += 1
            start = True
        elif k == 27:       # ESC
            cap1.release()
            cv2.destroyWindow('camera')
            return width, height, NULL_STRING

    while (d < MAX_FRAMES):
        # Capture frame-by-frame
        ret, frame_temp = cap1.read()
        frame_temp = cv2.resize(frame_temp, (0,0), frame2, scale, scale)
        
        # Display current camera frame
        cv2.imshow('camera',frame2)
        
        frame = (1.0*d/(d+1))*frame + frame2/(1.0*(d+1))
        d += 1
    
    cap1.release()
    cv2.destroyWindow('camera')

    return width, height, frame

def takePicture(fname, scale):
    w, h, f = takePic(fname, scale)
    if (f == NULL_STRING):
        return w, h
    #save frame
    cv2.imwrite('./data/'+fname+'.jpg', f)
    return w, h

def takeChessPicture(fname, scale):
    w, h, f = takePic(fname, scale)
    if (f == NULL_STRING):
        return w, h
    #save frame twice (because camcalib.py needs at least 2 images)
    cv2.imwrite('./data/'+fname+'.jpg', f)
    cv2.imwrite('./data/'+fname+'2.jpg', f)
    return w, h

def quickPic(fname, scale, npic):
    if scale <= 0:
        scale = 1
    
    start = False
    
    cap1 = cv2.VideoCapture(0)
    width = cap1.get(3)        # get width
    height = cap1.get(4)       # get height
    width, height = round(scale*width), round(scale*height)
    #width, height = np.rint(np.array([scale*width, scale*height]))

    ret, frame_temp = cap1.read()
    d = 0
    frame = np.zeros( (height, width, 3), np.uint8) # empty black image
    frame2 = np.zeros( (height, width, 3), np.uint8) # empty black image

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    
    # Wait for ' ' (space) to start taking pictures
    while (start == False):
        ret, frame_temp = cap1.read()
        cv2.resize(frame_temp, (0,0), frame, scale, scale)
        
        cv2.putText(frame, 'Press SPACE to start taking pictures of '+fname,
                    (int(width/10.0),int(height*9.0/10)),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255))
        cv2.imshow('camera',frame)
                    
        k = cv2.waitKey(10) & 0xFF
        if k == ord(' '):   # space
            d += 1
            start = True
        elif k == 27:       # ESC
            cap1.release()
            cv2.destroyWindow('camera')
            return width, height, NULL_STRING

    while (d < npic):
    # Capture frame-by-frame
        ret, frame_temp = cap1.read()
        frame_temp = cv2.resize(frame_temp, (0,0), frame2, scale, scale)
        
        # Display current camera frame
        cv2.imshow('camera',frame2)
        
        frame = (1.0*d/(d+1))*frame + frame2/(1.0*(d+1))
        d += 1
    
    cap1.release()
    cv2.destroyWindow('camera')

    print("frame width, height = "+str(frame.shape[1])+", "+str(frame.shape[0]))

    return width, height, frame

def quickPicture(fname, scale, npic):
    w, h, f = quickPic(fname, scale, npic)
    if (f == NULL_STRING):
        return w, h
    #save frame
    cv2.imwrite('./data/'+fname+'.jpg', f)
    return w, h




