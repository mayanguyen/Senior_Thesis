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

MAX_FRAMES = 100
NULL_STRING = 'null'

def takePic(fname, scale, npic):
    start = False
    d = 0
    
    # open captures for 2 cameras
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    
    # calculate new scaled dimensions for cap1
    width = cap1.get(3)        # get width
    height = cap1.get(4)       # get height
    width, height = round(scale*width), round(scale*height)
    
    # calculate scale2 for cap2;
    # cap1 and cap2 might have different sizes,
    # so different scale must be applied for cap2
    # to make it the same size as cap1
    scale_w = 1.*width/cap2.get(3)
    scale_h = 1.*height/cap2.get(4)
    scale2 = max(scale_w, scale_h)
    
    # empty black images
    frame1 = np.zeros( (height, width, 3), np.uint8) # empty black image
    frame2 = np.zeros( (height, width, 3), np.uint8) # empty black image
    frame3 = np.zeros( (height, width, 3), np.uint8) # empty black image
    frame4 = np.zeros( (height, width, 3), np.uint8) # empty black image
    preframe2 = np.zeros( (round(scale2*cap2.get(4)), round(scale2*cap2.get(3)), 3) , np.uint8)
    disp = np.zeros( (height, width*2, 3), np.uint8) # empty black image
    
    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    
    # Wait for ' ' (space) to start taking pictures
    while (start == False):
        # read & resize current frames of each camera
        ret, frame_temp1 = cap1.read()
        ret, frame_temp2 = cap2.read()
        cv2.resize(frame_temp1, (0,0), frame1, scale, scale)
        cv2.resize(frame_temp2, (0,0), preframe2, scale2, scale2)
        
        # if frame2 has different width:height ratio than frame1,
        # crop out the excess to make it the same size as frame1
        if (preframe2.shape[1] > width):
            diff = preframe2.shape[1] - width
            frame2 = preframe2[:, (diff/2):(width+(diff/2))]
        elif (preframe2.shape[0] > height):
            diff = preframe2.shape[0] - height
            frame2 = preframe2[(diff/2):(height+diff/2), :]
        else:
            frame2 = preframe2
        
        # concatenate and show the frames side-by-side
        disp[:, 0:width] = frame1
        disp[:, width:(2*width)] = frame2
        cv2.putText(disp, 'Press SPACE to start taking pictures of '+fname,
                    (int(width/10.0),int(height*9.0/10)),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255))
        cv2.imshow('camera', disp)
        
        # start taking pictures if SPACE is pressed
        # stop if ESC is pressed
        k = cv2.waitKey(10) & 0xFF
        if k == ord(' '):   # space
            d += 1
            start = True
        elif k == 27:       # ESC
            cap1.release()
            cap2.release()
            cv2.destroyWindow('camera')
            return width, height, NULL_STRING, None
    
    # take pics and average them (long-exposure effect)
    while (d < npic):
        # read & resize current frames of each camera
        ret, frame_temp1 = cap1.read()
        ret, frame_temp2 = cap2.read()
        cv2.resize(frame_temp1, (0,0), frame3, scale, scale)
        cv2.resize(frame_temp2, (0,0), preframe2, scale2, scale2)
        
        # if frame2 has different width:height ratio than frame1,
        # crop out the excess to make it the same size as frame1
        if (preframe2.shape[1] > width):
            diff = preframe2.shape[1] - width
            frame4 = preframe2[:, (diff/2):(width+(diff/2))]
        elif (preframe2.shape[0] > height):
            diff = preframe2.shape[0] - height
            frame4 = preframe2[(diff/2):(height+diff/2), :]
        else:
            frame4 = preframe2
    
        # concatenate and show the frames side-by-side
        disp[:, 0:width] = frame3
        disp[:, width:(2*width)] = frame4
        cv2.imshow('camera', disp)
        
        frame1 = (1.0*d/(d+1))*frame1 + frame3/(1.0*(d+1))
        frame2 = (1.0*d/(d+1))*frame2 + frame4/(1.0*(d+1))
        d += 1

    # release captures and destroy 'camera' window
    cap1.release()
    cap2.release()
    cv2.destroyWindow('camera')

    return width, height, frame1, frame2


def quickPicture(fname, version, scale, npic):
    w, h, f1, f2 = takePic(fname, scale, npic)
    if (f1 == NULL_STRING):
        return w, h
    #save frame
    cv2.imwrite('./data/test'+str(version)+'/'+fname+'1.jpg', f1)
    cv2.imwrite('./data/test'+str(version)+'/'+fname+'2.jpg', f2)
    
    return w, h

def takePicture(fname, version, scale):
    return quickPicture(fname, version, scale, MAX_FRAMES)

def takeChessPicture(fname, version, scale):
    w, h, f1, f2 = takePic(fname, scale, MAX_FRAMES)
    if (f1 == NULL_STRING):
        return w, h

    #save each frame twice (because camcalib.py needs at least 2 images)
    cv2.imwrite('./data/test'+str(version)+'/'+fname+'1.jpg', f1)
    cv2.imwrite('./data/test'+str(version)+'/'+fname+'_1.jpg', f1)
    cv2.imwrite('./data/test'+str(version)+'/'+fname+'2.jpg', f2)
    cv2.imwrite('./data/test'+str(version)+'/'+fname+'_2.jpg', f2)
    return w, h



#takePicture("test_2cameras", .25)

