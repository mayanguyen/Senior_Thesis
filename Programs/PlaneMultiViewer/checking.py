import os, sys
import numpy as np
import cv2
import longexposure as le

camera_scale = .25
#le.takePicture('check_anamorph', 4, camera_scale)
le.takePicture('check_anamorphSingleTwo', 5, camera_scale)