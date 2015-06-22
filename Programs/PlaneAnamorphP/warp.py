import numpy as np
import cv2

'''backwarp not tested'''
# Backward warping: loop through anamorph image (resulting pre-warped image)
def backbarp(img, h):
    morph = np.zeros((proj_height, proj_width, 3), np.uint8) # blank image
    for y in range(morph.shape[0]):
        for x in range(morph.shape[1]):
            p = np.array([[x],[y],[1]]) #([[x+left],[y+top],[1]])
            p = np.mat(p) # vertical
            p = h.dot(p)
            p /= p[2]
            
            if 0 < int(p[0][0]) < img.shape[1] and 0 < int(p[1][0]) < img.shape[0]:
                morph[y][x] = img[int(p[1][0])][int(p[0][0])]
    return morph

#forward warp
def forwarp(img, hInv):
    morph = np.zeros((proj_height, proj_width, 3), np.uint8) # blank image
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            p = np.array([[x],[y],[1]]) #([[x+left],[y+top],[1]])
            p = np.mat(p) # vertical
            p = hInv.dot(p)
            p /= p[2]
            
            if 0 < int(p[0][0]) < morph.shape[1] and 0 < int(p[1][0]) < morph.shape[0]:
                morph[int(p[1][0])][int(p[0][0])] = img[y][x]
    return morph


