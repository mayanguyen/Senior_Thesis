## Plane Anamorphism

'''
    Author: Van Mai Nguyen Thi
    Senior Project
    October 2014 - January 2015

    1. Calibrate:
        a. Project a rectangle
        b. Camera: detect corners
        c. Generate homography H relating camera image and projector image
    2. Using the H, (backward) warp the image to create the anamorphic image
    3. Projector: display the resulting image
    
    
Reference:
 - Video capture: from 'collect.py', from moodle2,bard.edu posted by Keith O'Hara.

'''

import numpy as np
import cv2
import longexposure as le
import camcalib as cc
# import warp # my implementations of wackwarp and forwarp

'''delete: width = 848# proj.shape[1]
height = 480# proj.shape[0]'''

# Sort corners: top/bottom-left/right
# Get 2 leftmost, then 2 rightmost
# and among them find higher and lower corner
# Order of corners: from top-left clockwise
# i.e. top-left, top-right, bottom-right, bottom-left
def sortCorners(c):
    c2 = np.zeros((4, 2), np.float32)
    minx1 = cam_width
    minx2 = cam_width
    idx1 = 0
    idx2 = 0
    for i in range(4):
        if c[i][0][0] < minx1:
            minx2 = minx1
            idx2 = idx1
            minx1 = c[i][0][0]
            idx1 = i
        elif c[i][0][0] < minx2:
            minx2 = c[i][0][0]
            idx2 = i

    if c[idx1][0][1] <= c[idx2][0][1]:
        c2[0] = c[idx1][0]
        c2[3] = c[idx2][0]
    else:
        c2[0] = c[idx2][0]
        c2[3] = c[idx1][0]

    idx3 = 10
    idx4 = 0
    for i in range(4):
        if i != idx1 and i != idx2:
            if idx3 == 10:
                idx3 = i
            else:
                idx4 = i

    if c[idx3][0][1] <= c[idx4][0][1]:
        c2[1] = c[idx3][0]
        c2[2] = c[idx4][0]
    else:
        c2[1] = c[idx4][0]
        c2[2] = c[idx3][0]
    return c2

# return largest rectangle (as list of corners from top-left clockwise)
# that fits inside the shape given
# input shape = list of 4 corners of the shape
def maxRect(shape):
    shape2 = shape
    
    shape2[0] = [max(shape[0][0], shape[3][0]), max(shape[0][1], shape[1][1])]
    shape2[1] = [min(shape[1][0], shape[2][0]), max(shape[0][1], shape[1][1])]
    shape2[2] = [min(shape[1][0], shape[2][0]), min(shape[2][1], shape[3][1])]
    shape2[3] = [max(shape[0][0], shape[3][0]), min(shape[2][1], shape[3][1])]
    
    return shape2


########################### Take pictures of the screen and projection

camera_scale = .4
cam_width, cam_height = le.takePicture('screen', camera_scale)
le.takeChessPicture('chessProjection', camera_scale)
print('Width, height of cam image = '+str(cam_width)+','+str(cam_height))

screen_width, screen_height = cam_width, cam_height # width and height of poster/screen
''' change this: to reflect the actual size of the poster '''

chesspoints = cc.camCalib()

########################### Load Images & Set Variables
#delete this: proj = cv2.imread('./data/blank.jpg')
#screen = cv2.imread('./data/screen.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
#scr = cv2.imread('./data/screen.jpg')
screen = cv2.imread('./data/screen.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
#screen_corners = cv2.imread('./data/screen.jpg')
chessprojection = cv2.imread('./data/chessProjection.jpg')
#image = cv2.imread('./data/small_rect.jpg') #pic to be warped
image = cv2.imread('./data/lighthouse.jpg') #pic to be warped
chessboard = cv2.imread('./data/chessboard.jpg') #just 6x9 chessboard

proj_width, proj_height = chessboard.shape[1], chessboard.shape[0]


########################### Detect Corners of the Screen
# I used goodFeaturesToTrack() because it is easier to get the top 4 corners,
# as well as specify the minimum distance between each feature

corners = cv2.goodFeaturesToTrack(screen, 4, 0.01, screen.shape[0]/10)
corners = sortCorners(corners)

print('Detected corners: '+str(corners))
for i in range(4):
    cv2.circle(screen,
        (int(corners[i][0]),int(corners[i][1])), 10, (255,0,0), -1)

cv2.imshow('screen_corners',screen)
cv2.imwrite('./data/screen_corners.jpg', screen)

########################### Generate homography H_SC: screen -> camera
'''TODO measure the screen/poster width/height ratio'''
src_corners = np.array([
    [0.,0.],
    [screen_width-1, 0.],
    [screen_width-1, screen_height-1],
    [0., screen_height-1]], np.float32)
print('src_corners = '+str(src_corners))

H_SC, mask = cv2.findHomography(src_corners, corners, cv2.RANSAC)
#print('H_SC = '+str(H_SC))

HInv_SC = np.linalg.inv(H_SC)
#print('\nHInv_SC (inverse) = '+str(HInv_SC))

########################### Generate homography H_PC: projector -> camera
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(4,7,0) (?)
src_corners = np.zeros((5*8,2), np.float32)
src_corners[:,:] = np.mgrid[2:10,2:7].T.reshape(-1,2)
w = min(chessboard.shape[0]/8, chessboard.shape[1]/11)
src_corners *= w
print('\nProjector src_corners = '+str(src_corners))

H_PC, mask = cv2.findHomography(src_corners, chesspoints, cv2.RANSAC, 5.0)
print('H_PC = '+str(H_PC))

HInv_PC = np.linalg.inv(H_PC)

########################### Prepare image to be warped

### Find boundaries of the projection in the camera image
projector_corners = np.float32([ [0,0],
                               [11*w-1, 0],
                               [11*w-1, 8*w-1],
                               [0, 8*w-1] ]).reshape(-1,1,2)
print('projector corners = '+str(projector_corners))

projectionInCamera_corners = cv2.perspectiveTransform(projector_corners, H_PC)
for i in range(4):
    cv2.circle(chessprojection,
    (int(projectionInCamera_corners[i][0][0]),int(projectionInCamera_corners[i][0][1])), 10, (255,0,0), -1)
cv2.imshow('projection_corners', chessprojection)
cv2.imwrite('./data/projectionInCameraCorners.jpg', chessprojection)

### Find the largest rectangle that fits inside the projection
max_rect = maxRect(sortCorners(projectionInCamera_corners))
chessprojection = cv2.imread('./data/chessProjection.jpg')
for i in range(4):
    cv2.circle(chessprojection,
               (int(max_rect[i][0]),int(max_rect[i][1])), 10, (255,0,0), -1)
cv2.imshow('max_rect', chessprojection)
cv2.imwrite('./data/max_rect.jpg', chessprojection)
print('projectionInCamera corners = '+str(projectionInCamera_corners))
print('max rect = '+str(max_rect))


''' ^ tested 'til here '''


## Corners of max_rect must have positive coordinates because we assume that
## the camera can see the entire screen (which contains the entire projection)

rect_w, rect_h = max_rect[1][0]-max_rect[0][0]+1, max_rect[2][1]-max_rect[1][1]+1
image_w, image_h = image.shape[1], image.shape[0]
scale = min(1.*rect_w/image_w, 1.*rect_h/image_h)

image_w, image_h = int(image_w*scale), int(image_h*scale)
image = cv2.resize(image, (image_w, image_h))
pre_anamorph = np.zeros((rect_h+max_rect[0][1],
                         rect_w+max_rect[0][0], 3), np.uint8) # empty black image
pre_anamorph[max_rect[0][1]:(max_rect[0][1]+image_h),
             max_rect[0][0]:(max_rect[0][0]+image_w)] = image

cv2.imshow('pre-anamorph', pre_anamorph)
cv2.imwrite('./data/pre_anamorph.jpg', pre_anamorph)

''' seems like the pre-anamorph is correct '''

########################### Warp (create anamorphic image)
#anamorph = warp.forwarp(image, HInv_PC)
anamorph = cv2.warpPerspective(pre_anamorph, HInv_PC,
                               (chessboard.shape[1],chessboard.shape[0]))

print('anamorph height,width = '+str(anamorph.shape[0])+','+str(anamorph.shape[1]))

########################### Show & save images
#cv2.imshow('image', image)
cv2.imshow('anamorph', anamorph)
cv2.imwrite('./data/anamorph.jpg', anamorph)
cv2.imwrite('./displayFullScreen/data/anamorph.jpg', anamorph)

print('Created anamorph')

le.takePicture('check_anamorph', camera_scale)

###########################

k = cv2.waitKey(0) & 0xFF
if k == 27 or k == ord('q'):         # wait for ESC or 'q' key to exit
    cv2.destroyAllWindows()


