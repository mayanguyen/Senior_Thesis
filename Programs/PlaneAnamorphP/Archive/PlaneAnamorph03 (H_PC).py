## Plane Anamorphism

'''
    Author: Van Mai Nguyen Thi
    Senior Project
    October 2014 - November 2014

    1. Calibrate:
        a. Project a rectangle
        b. Camera: detect corners
        c. Generate homography H relating camera image and projector image
    2. Using the H, (backward) warp the image to create the anamorphic image
    3. Projector: display the resulting image
    
    
Reference:
 - Video capture: from 'colect.py', from moodle2,bard.edu posted by Keith O'Hara.

'''

import numpy as np
import cv2

proj1 = cv2.imread('./data/blank.jpg')
width = proj1.shape[1]
height = proj1.shape[0]
cam1 = cv2.imread('./data/long_exposure.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
dst = cv2.imread('./data/long_exposure.jpg')
#cam2 = cv2.imread('./data/long_exposure.jpg') #pic to be warped
cam2 = cv2.imread('./data/small_rect.jpg') #pic to be warped

########################### Detect Corners
# I used goodFeaturesToTrack() because it is easier to get the top 4 corners,
# as well as specify the minimum distance between each feature

corners = cv2.goodFeaturesToTrack(cam1, 4, 0.01, cam1.shape[0]/20) # this is 3D!

########################### Classify corners: top/bottom-left/right
# Get 2 leftmost, then among them get higher and lower corner
# Order of corners: from top-left clockwise
#   i.e. top-left, top-right, bottom-right, bottom-left

corners2 = corners
corners = np.empty((4,2)) # 2D array: 4 rows, 2 columns
minx1 = cam1.shape[1] # width
minx2 = cam1.shape[1] # width
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
for i in range(4):
    cv2.circle(dst, (int(corners[i][0]),int(corners[i][1])), 10, (255,0,0), -1)

cv2.imshow('dst',dst)

########################### Generate homography H: projector -> camera
### based on Keith's lab about homographies

src_corners = np.array([[0.0,0.0], [width-1,0.0], [width-1,height-1], [0.0,height-1]])

H, mask = cv2.findHomography(src_corners, corners, cv2.RANSAC,5.0)
print('H = '+str(H))

HInv = np.linalg.inv(H)
print('\nHInv (inverse) = '+str(HInv))

# Verify that H is correct by checking that src_corners map to corners
for i in range(4):
    p = np.array([[src_corners[i][0]], [src_corners[i][1]],[1]])
    p = np.mat(p) # vertical
    p = H.dot(p)
    p /= p[2]
    
    if p[0][0] == corners[i][0] and p[1][0] == corners[i][1]:
        print(str(p)+'='+str(corners[i]))
    else:
        print(str(p)+'NOT ='+str(corners[i]))

# Verify that HInv is correct
print 'Check HInv:'
for i in range(4):
    p = np.array([[corners[i][0]], [corners[i][1]],[1]])
    p = np.mat(p) # vertical
    p = HInv.dot(p)
    p /= p[2]
    
    if p[0][0] == src_corners[i][0] and p[1][0] == src_corners[i][1]:
        print(str(p)+'='+str(src_corners[i]))
    else:
        print(str(p)+'NOT ='+str(src_corners[i]))

########################### Warp (create anamorphic image)
'''H = np.array([[np.cos(np.pi/10), -np.sin(np.pi/10), 0],
              [np.sin(np.pi/10),  np.cos(np.pi/10), 0],
              [0, 0, 1] ])'''

# Check where corners will land to estimate best size for anamorph image
'''pts = np.array([[0.0, 0.0, 1],
                [cam2.shape[1], 0.0, 1],
                [cam2.shape[1], cam2.shape[0], 1],
                [0.0, cam2.shape[0], 1]])

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

anamorph = np.zeros((int(bottom-top+1),int(right-left+1),3), np.uint8) # blank image
#np.zeros((height,width, 3), np.uint8)'''

anamorph = np.zeros((height,width,3), np.uint8) # blank image

print('anamorph.shape = '+str(anamorph.shape[0])+','+str(anamorph.shape[1]))

# Backward warping: loop through anamorph image (resulting pre-warped image)
# src = stored in variable cam2; this is the image image we want to see in the camera
# dst = anamorph; pre-warped version of src
'''for y in range(anamorph.shape[0]):
    for x in range(anamorph.shape[1]):
        p = np.array([[x],[y],[1]]) #([[x+left],[y+top],[1]])
        p = np.mat(p) # vertical
        p = H.dot(p)
        p /= p[2]
        
        if 0 < int(p[0][0]) < cam2.shape[1] and 0 < int(p[1][0]) < cam2.shape[0]:
            anamorph[y][x] = cam2[int(p[1][0])][int(p[0][0])]'''

#forward warp
for y in range(cam2.shape[0]):
    for x in range(cam2.shape[1]):
        p = np.array([[x],[y],[1]]) #([[x+left],[y+top],[1]])
        p = np.mat(p) # vertical
        p = HInv.dot(p)
        p /= p[2]
        
        if 0 < int(p[0][0]) < anamorph.shape[1] and 0 < int(p[1][0]) < anamorph.shape[0]:
            anamorph[int(p[1][0])][int(p[0][0])] = cam2[y][x]
        '''if 0 < int(p[0][0]-left) < anamorph.shape[1] and 0 < int(p[1][0]-top) < anamorph.shape[0]:
            anamorph[int(p[1][0]-top)][int(p[0][0]-left)] = cam2[y][x]'''

cv2.imshow('cam2',cam2)
cv2.imshow('anamorph',anamorph)
cv2.imwrite('anamorph.jpg', anamorph)
cv2.imwrite('./displayFullScreen/data/anamorph.jpg', anamorph)

print('Created anamorph')

###########################

k = cv2.waitKey(0) # & 0xFF
if k == 27 or k == ord('q'):         # wait for ESC or 'q' key to exit
    cv2.destroyAllWindows()




