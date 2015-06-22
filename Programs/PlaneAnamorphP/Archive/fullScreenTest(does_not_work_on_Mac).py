import numpy as np
import cv2



def main():
    img = cv2.imread('./data/small_rect.jpg')

    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    cv2.imshow('frame',img)
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()

main()