from PIL.Image import *
import numpy as np
import cv2

img = cv2.imread('color.png')
#cv2.imwrite('myImage_RGB.png', img)
#img_hsv = convert_colorspace(img, 'RGB', 'HSV')
img = img[:,:,::-1]
cv2.imwrite('myImage.png', img)