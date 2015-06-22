import numpy as np
import cv2
import time

MAX_FRAMES = 100
NULL_STRING = 'null'

def pic(fname, scale, width, height, npic, camID):
    start = False
    d = 0
    
    cap1 = cv2.VideoCapture(camID)
    
    if scale <= 0:
        scale = width/cap1.get(3)
        if (scale != height/cap1.get(4)):
            print("Incorrect ratio. scale_w, scale_h = "+str(scale)+", "+str(height/cap1.get(4)))
    else:
        width = cap1.get(3)        # get width
        height = cap1.get(4)       # get height
        width, height = round(scale*width), round(scale*height)

    ret, frame_temp = cap1.read()
    
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
    
    print("frame width, height = "+str(frame.shape[1])+", "+str(frame.shape[0]))

    cap1.release()
    cv2.destroyWindow('camera')
    return width, height, frame

def takePic(fname, scale, npic):
    w, h, f = pic(fname, scale, 0, 0, npic, 0)
    if (f == NULL_STRING):
        return w, h
    #save frame
    cv2.imwrite('./data/'+fname+'1.jpg', f)

    w, h, f = pic(fname, scale, 768, 432, npic, 1)
    if (f == NULL_STRING):
        return w, h
    #save frame
    cv2.imwrite('./data/'+fname+'2.jpg', f)

    return w, h

takePic("test_2cameras", .4, 2)