'''
This file was downloaded from moodle2.bard.edu, Computational Image course.
By Keith O'Hara?

Modified slightly by Van Mai Nguyen Thi

relevant opencv documentation:
http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
'''

import numpy as np
import cv2

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3,width)  # set width
cap.set(4,height)  # set height

d = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord(' '):
        print("frame-"+str(d))
        cv2.imwrite('frame-%d.jpg' %d, gray)
        d += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
