## Plane Anamorphism
## 2 viewers

'''
    Author: Van Mai Nguyen Thi
    Senior Project
    February 2015

    Method 1.5: put chess_corners1 and chess_corners2 into findHomography(), and get sth like the average H_PC

'''

import os, sys
import numpy as np
import cv2
import longexposure as le
import chesscorners as cc
import leastSquares as ls
import maxrect as mr
# import warp # my implementations of wackwarp and forwarp


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
'''def maxRect(shape):
    shape2 = shape
    
    shape2[0] = [max(shape[0][0], shape[3][0]), max(shape[0][1], shape[1][1])]
    shape2[1] = [min(shape[1][0], shape[2][0]), max(shape[0][1], shape[1][1])]
    shape2[2] = [min(shape[1][0], shape[2][0]), min(shape[2][1], shape[3][1])]
    shape2[3] = [max(shape[0][0], shape[3][0]), min(shape[2][1], shape[3][1])]
    
    return shape2
'''

def fitInside(shape, limitShape):
    for i in range(4):
        if (shape[i][0][0] < 0):
            shape[i][0][0] = 0
        elif (shape[i][0][0] >= limitShape[1]):
            shape[i][0][0] = limitShape[1]-1
        if (shape[i][0][1] < 0):
            shape[i][0][1] = 0
        elif (shape[i][0][1] >= limitShape[0]):
            shape[i][0][1] = limitShape[0]-1
    return shape

def findCorners(img):
    corners = cv2.goodFeaturesToTrack(img, 4, 0.01, img.shape[0]/10)
    corners = sortCorners(corners)
    print('Detected corners: '+str(corners))
    for i in range(4):
        cv2.circle(img, (int(corners[i][0]),int(corners[i][1])), 10, (255,0,0), -1)
    return img, corners


def draw(points, imgPath):
    img = cv2.imread(imgPath)
    for i in range(points.shape[0]):
        cv2.circle(img, (int(points[i][0]),int(points[i][1])), 5, (0,0,255), -1)
    return img

######## Generate correct version, so that files are saved in a new folder.

f = open('naming.txt', 'r')

version = int(f.readline())
version = version+1
f.close()

f = open('naming.txt', 'w')
f.write(str(version))
f.close()

dir = './data/test'+str(version)
if not os.path.exists(dir):
    os.makedirs(dir);

print('Running test version #'+str(version))



########################### Take pictures of the screen and projection
camera_scale = .25
cam_width, cam_height = le.takePicture('screen', version, camera_scale)
le.takeChessPicture('chessProjection', version, camera_scale)
#cam_width, cam_height = 648, 486 #NoCam
print('Width, height of cam image = '+str(cam_width)+','+str(cam_height))

screen_width, screen_height = cam_width, cam_height # width and height of poster/screen
''' change this: to reflect the actual size of the poster '''

chess_corners1, chess_corners2 = cc.chessCorners(version)



########################### Load Images & Set Dimension Variables
#chessprojection1 = cv2.imread('./data/test'+str(version)+'/chessProjection1.jpg')
#GOOD: 
image = cv2.imread('./data/lighthouse.jpg') #pic to be warped
#image = cv2.imread('./data/chessboard.jpg') #pic to be warped

chessboard = cv2.imread('./data/chessboard.jpg') #just 6x9 chessboard

proj_width, proj_height = chessboard.shape[1], chessboard.shape[0]



########################### Detect Corners of the Screen
# I used goodFeaturesToTrack() because it is easier to get the top 4 corners,
# as well as specify the minimum distance between each feature

screen = cv2.imread('./data/test'+str(version)+'/screen1.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
screen, screen_corners1 = findCorners(screen)
cv2.imshow('screen_corners1', screen)
cv2.imwrite('./data/test'+str(version)+'/screen_corners1.jpg', screen)

screen = cv2.imread('./data/test'+str(version)+'/screen2.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
screen, screen_corners2 = findCorners(screen)
cv2.imshow('screen_corners2', screen)
cv2.imwrite('./data/test'+str(version)+'/screen_corners2.jpg', screen)



########################### Generate homography H_SC: screen -> camera
'''TODO measure the screen/poster width/height ratio'''
src_corners = np.array([
    [0.,0.],
    [screen_width-1, 0.],
    [screen_width-1, screen_height-1],
    [0., screen_height-1]], np.float32)
print('src_corners = '+str(src_corners))

H1_SC, mask = cv2.findHomography(src_corners, screen_corners1, cv2.RANSAC)
H2_SC, mask = cv2.findHomography(src_corners, screen_corners2, cv2.RANSAC)
#print('H_SC = '+str(H_SC))

H1Inv_SC = np.linalg.inv(H1_SC)
H2Inv_SC = np.linalg.inv(H2_SC)
#print('\nHInv_SC (inverse) = '+str(HInv_SC))

########################### Generate AVERAGE homography H_PC: projector -> camera
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(4,7,0) (?)
src_corners1 = np.zeros((5*8,2), np.float32)
src_corners1[:,:] = np.mgrid[2:10,2:7].T.reshape(-1,2)
w = min(chessboard.shape[0]/8, chessboard.shape[1]/11)
src_corners1 *= w
src_corners = np.concatenate((src_corners1, src_corners1), axis=0)
print('\n2x Projector src_corners = '+str(src_corners))

chess_corners = np.concatenate((chess_corners1, chess_corners2), axis=0)

H_hat, mask = cv2.findHomography(src_corners, chess_corners, cv2.RANSAC, 5.0)
H_hatInv = np.linalg.inv(H_hat)

'''

### Testing on each camera individually
src_corners = np.zeros((5*8,2), np.float32)
src_corners[:,:] = np.mgrid[2:10,2:7].T.reshape(-1,2)
w = min(chessboard.shape[0]/8, chessboard.shape[1]/11)
src_corners *= w
#src_corners = np.concatenate((src_corners1, src_corners1), axis=0)

#chess_corners = np.concatenate((chess_corners1, chess_corners2), axis=0)

H_PC1, mask = cv2.findHomography(src_corners, chess_corners1, cv2.RANSAC, 5.0)
HInv_PC1 = np.linalg.inv(H_PC1)
H_PC2, mask = cv2.findHomography(src_corners, chess_corners2, cv2.RANSAC, 5.0)
HInv_PC2 = np.linalg.inv(H_PC2)

print("H_PC1 = \n"+str(H_PC1))
print("H_PC2 = \n"+str(H_PC2))


q1 = ls.LSChessboard(chess_corners1, 5, 8)
q2 = ls.LSChessboard(chess_corners2, 5, 8)

# draw new chessboard on cam images (new targets)
newChess1 = draw(q1, './data/test'+str(version)+'/chessProjection1.jpg')
newChess2 = draw(q2, './data/test'+str(version)+'/chessProjection2.jpg')
cv2.imwrite('./data/test'+str(version)+'/newChess1.jpg', newChess1)
cv2.imwrite('./data/test'+str(version)+'/newChess2.jpg', newChess2)

H_hatInv = ls.HhatInv(H_PC1, H_PC2, src_corners, q1, q2)
H_hat = np.linalg.inv(H_hatInv)

#print('q1 = \n'+str(q1))
#print('q2 = \n'+str(q2))
print('H_hatInv = \n'+str(H_hatInv))
'''

########################### Prepare image to be warped

### Find boundaries of the projection in the orig image
projCorners = np.float32([ [0,0],
                           [chessboard.shape[1]-1, 0],
                           [chessboard.shape[1]-1, chessboard.shape[0]-1],
                           [0, chessboard.shape[0]-1] ]).reshape(-1,1,2)
#                           [11*w-1, 0],
#                           [11*w-1, 8*w-1],
#                           [0, 8*w-1] ]).reshape(-1,1,2)
print('projector corners = '+str(projCorners))

#chessprojection1 = cv2.imread('./data/test'+str(version)+'/chessProjection1.jpg')
projCornersInOrig = cv2.perspectiveTransform(projCorners, H_hat)
'''for i in range(4):
    cv2.circle(chessprojection1,
    (int(projectionInCamera_corners[i][0][0]),int(projectionInCamera_corners[i][0][1])), 10, (255,0,0), -1)
cv2.imshow('projection_corners', chessprojection1)
cv2.imwrite('./data/test'+str(version)+'/projectionInCameraCorners.jpg', chessprojection1)'''

# Do I need this???
projCornersInOrig = sortCorners(projCornersInOrig)
#sortCorners(fitInside(projCornersInOrig, chessboard.shape))

### Find the largest rectangle that fits inside the projection
max_rect = mr.maxRect(screen.shape[1], screen.shape[0], projCornersInOrig, image.shape[1], image.shape[0])
'''chessprojection1 = cv2.imread('./data/test'+str(version)+'/chessProjection1.jpg')
for i in range(4):
    cv2.circle(chessprojection1,
               (int(max_rect[i][0]),int(max_rect[i][1])), 10, (255,0,0), -1)
cv2.imshow('max_rect', chessprojection1)
cv2.imwrite('./data/test'+str(version)+'/max_rect.jpg', chessprojection1)'''
print('projCornersInOrig = '+str(projCornersInOrig))
print('max rect = '+str(max_rect))

## Corners of max_rect must have positive coordinates because we assume that
## the camera can see the entire screen (which contains the entire projection)

rect_w, rect_h = max_rect[1][0]-max_rect[0][0]+1, max_rect[2][1]-max_rect[1][1]+1
image_w, image_h = image.shape[1], image.shape[0]
scale = min(1.*rect_w/image_w, 1.*rect_h/image_h)

image_w, image_h = int(image_w*scale), int(image_h*scale)
print("image_w, h = "+str(image_w)+", "+str(image_h))
image = cv2.resize(image, (image_w, image_h))
#pre_anamorph = np.zeros((rect_h+max_rect[0][1],rect_w+max_rect[0][0], 3), np.uint8) # empty black image
pre_anamorph = np.zeros((screen.shape[0], screen.shape[1], 3), np.uint8) # empty black image
pre_anamorph[max_rect[0][1]:(max_rect[0][1]+image_h),
             max_rect[0][0]:(max_rect[0][0]+image_w)] = image

cv2.imshow('pre-anamorph', pre_anamorph)
cv2.imwrite('./data/test'+str(version)+'/pre_anamorph.jpg', pre_anamorph)



########################### Warp (create anamorphic image)
anamorph = cv2.warpPerspective(pre_anamorph,H_hatInv,(chessboard.shape[1],chessboard.shape[0]))

print('anamorph height,width = '+str(anamorph.shape[0])+','+str(anamorph.shape[1]))



########################### Simulator check



########################### Show & save images
#cv2.imshow('image', image)
cv2.imshow('anamorph', anamorph)
cv2.imwrite('./data/test'+str(version)+'/anamorph.jpg', anamorph)
cv2.imwrite('./displayFullScreen/data/anamorph.jpg', anamorph)

print('Created anamorph')

le.takePicture('check_anamorph', version, camera_scale)

###########################

k = cv2.waitKey(0) & 0xFF
if k == 27 or k == ord('q'):         # wait for ESC or 'q' key to exit
    cv2.destroyAllWindows()


