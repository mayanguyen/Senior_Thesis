## Plane Anamorphism

'''
    Author: Van Mai Nguyen Thi
    Senior Project
    October 2014

    1. Calibrate:
+       a. Project a rectangle
+       b. Camera: detect corners
        c. Generate homography H relating camera image and projector image
    2. Using the HInv, warp the image to create the anamorphic image
    3. Projector: display the resulting image
    
    
Sources:
 - Video capture: from 'colect.py', from moodle2,bard.edu posted by Keith O'Hara.

'''

import numpy as np
import cv2

width = 640
height = 480

pic = cv2.imread('blank.jpg') #pic to be warped

img = cv2.imread('frame-0.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
dst = cv2.imread('frame-0.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #np.zeros((height,width,3), np.uint8)
#dst = cv2.imread('destination.jpg')
'''cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
cv2.imshow('warped',img)'''

###########################
#cv2.cornerHarris(img, 2, 3, 0.04, dst, cv2.BORDER_CONSTANT)
'''dst = cv2.cornerHarris(img,5,9,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',dst)'''

########################### Detect Corners
# I used goodFeaturesToTrack() because it is easier to get the top 4 corners,
# as well as specify the minimum distance between each feature
''' http://docs.opencv.org/modules/imgproc/doc/feature_detection.html '''
''' http://docs.opencv.org/modules/core/doc/drawing_functions.html '''
corners = cv2.goodFeaturesToTrack(img, 4, 0.01, height/20) # this is 3D!

for i in range(4):
    cv2.circle(dst, (corners[i][0][0],corners[i][0][1]), 10, (255,0,0), -1)

cv2.imshow('dst',dst)

########################### Classify corners: top/bottom-left/right
# Get 2 leftmost, then among them get higher and lower corner
# Order of corners: from top-left clockwise
#   i.e. top-left, top-right, bottom-right, bottom-left

corners2 = corners
corners = np.empty((4,2)) # 2D array: 4 rows, 2 columns
minx1 = width
minx2 = width
idx1 = 0
idx2 = 0
for i in range(4):
    if corners2[i][0][0] < minx1:
        minx2 = minx1
        idx2 = idx1
        minx1 = corners2[i][0][0]
        idx1 = i
    elif corners2[i][0][0] < minx2:
        minx2 = corners2[i][0][0]
        idx2 = i

if corners2[idx1][0][1] <= corners2[idx2][0][1]:
    corners[0] = corners2[idx1][0]
    corners[3] = corners2[idx2][0]
else:
    corners[0] = corners2[idx2][0]
    corners[3] = corners2[idx1][0]

idx3 = 10
idx4 = 0
for i in range(4):
    if i != idx1 and i != idx2:
        if idx3 == 10:
            idx3 = i
        else:
            idx4 = i

if corners2[idx3][0][1] <= corners2[idx4][0][1]:
    corners[1] = corners2[idx3][0]
    corners[2] = corners2[idx4][0]
else:
    corners[1] = corners2[idx4][0]
    corners[2] = corners2[idx3][0]

print('Detected corners: '+str(corners))

########################### Homography test
''' http://docs.scipy.org/doc/numpy/reference/routines.array-creation.html '''
'''H = np.mat(((0.004,-0.001,0.769),(-0.001,0.004,0.639),(0,0,0.012)))
#H = H*(1/0.012)
for y in range(height):
    for x in range(width):
        p = np.array([[x],[y],[1]])
        p = np.mat(p) # vertical
        p = H*p
        #if p[0][0] < anamorph
        #anamorph[y][x] = pic[y][x]
        if 0 < p[1][0]/p[2][0] < height and 0 < p[0][0]/p[2][0] < width:
            anamorph[int(p[1][0]/p[2][0])][int(p[0][0]/p[2][0])] = pic[y][x]

cv2.imshow('anamorph',anamorph)'''



########################### Generate homography H: projector image -> camera image
### based on Keith's lab about homographies
src_corners = np.array([[0.0,0.0], [width-1,0.0], [width-1,height-1], [0.0,height-1]])

H, mask = cv2.findHomography(src_corners, corners, cv2.RANSAC,5.0)
#matchesMask = mask.ravel().tolist()

print('H = '+str(H))

#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#dst = cv2.perspectiveTransform(pts,M)


'''a = np.empty((8,9))
for i in range(4):
    x = src_corners[i][0]
    y = src_corners[i][1]
    X = corners[i][0]
    Y = corners[i][1]
    l1 = [X, Y, 1, 0, 0, 0, -X*x, -Y*x, -x]
    l2 = [0, 0, 0, X, Y, 1, -X*y, -Y*y, -y]
    a[2*i] = l1
    a[2*i+1] = l2


A = np.mat(a)
print(A)
A = A.transpose()*A
values, vectors = np.linalg.eig(A)

print('values = '+str(values))
print('vectors = '+str(vectors))

mineigIdx = 0
for i in range(values.size):
    if (values[i] < values[mineigIdx]):
        mineigIdx = i

print('min eig = '+str(values[mineigIdx]))

H = vectors[mineigIdx].reshape((3,3))
print('H = '+str(H))'''

########################### Get H^{-1}: camera image -> projector image
'''H = np.array([[np.cos(np.pi/10), -np.sin(np.pi/10), 0],
              [np.sin(np.pi/10),  np.cos(np.pi/10), 0],
              [0, 0, 1] ])'''

HInv = np.linalg.inv(H)
print('HInv (inverse) = '+str(HInv))

# Check where corners will land to estimate best size for anamorph image
pts = np.array([[0.0, 0.0, 1],
                [pic.shape[0], 0.0, 1],
                [pic.shape[0], pic.shape[1], 1],
                [0.0, pic.shape[1], 1]])
'''([[pic.shape[0]/4.0,   pic.shape[1]/4.0,   1],
                [pic.shape[0]*3/4.0, pic.shape[1]/4.0,   1],
                [pic.shape[0]*3/4.0, pic.shape[1]*3/4.0, 1],
                [pic.shape[0]/4.0,   pic.shape[1]*3/4.0, 1]])'''
print('Corners of the picture to be warped: \n'+str(pts))
pts = HInv.dot(pts.transpose())
print('Corners projection on anamorph image: \n'+str(pts))
pts /= pts[2]
pts = pts.transpose()
print('Corners projection on anamorph image (scaled for w=1): \n'+str(pts))
left    = min(pts[0][0], pts[3][0])
right   = max(pts[1][0], pts[2][0])
top     = min(pts[0][1], pts[1][1])
bottom  = max(pts[2][1], pts[3][1])

anamorph = np.zeros((int(right-left+2),int(bottom-top+2),3), np.uint8) # blank image


#anamorph = np.zeros((2*height,2*width,3), np.uint8) # blank image
print('anamorph.shape = '+str(anamorph.shape))

'''for y in range(height):#[0,height-1]:
    for x in range(width):#[0,width-1]:
        p = np.array([[x],[y],[1]])
        p = np.mat(p) # vertical
        #print('p = '+str(p))
        p = HInv.dot(p)
        #print('HInv.dot(p) = '+str(p))
        
        #if p[0][0] < anamorph
        #anamorph[y][x] = pic[y][x]

        if 0 < p[1][0]/p[2][0] < anamorph.shape[1] and 0 < p[0][0]/p[2][0] < anamorph.shape[0]:
            #print ('color = '+str(pic[y][x]))
            anamorph[int(p[1][0]/p[2][0])][int(p[0][0]/p[2][0])] = pic[y][x]'''


# backward warping: loop through anamorph image (resulting pre-warped image)
# because we don't want gaps
for y in range(anamorph.shape[1]):
    for x in range(anamorph.shape[0]):
        p = np.array([[x+left+1],[y+top+1],[1]])
        p = np.mat(p) # vertical
        #print('p = '+str(p))
        p = H.dot(p)
        p /= p[2]
        #print('HInv.dot(p) = '+str(p))
        
        #if p[0][0] < anamorph
        #anamorph[y][x] = pic[y][x]
        
        if 0 < int(p[0][0]) < pic.shape[0] and 0 < int(p[1][0]) < pic.shape[1]:
            #print ('color = '+str(pic[y][x]))
            anamorph[x][y] = pic[int(p[0][0])][int(p[1][0])]


cv2.imshow('pic',pic)
cv2.imshow('anamorph',anamorph)
cv2.imwrite('anamorph.jpg', anamorph)





###########################

k = cv2.waitKey(0) # & 0xFF
if k == 27 or k == ord('q'):         # wait for ESC or 'q' key to exit
    cv2.destroyAllWindows()



'''a = np.arange(10)
a[0]
for i in range(10):
    print(a[i])'''

