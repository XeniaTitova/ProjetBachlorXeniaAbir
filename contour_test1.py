import cv2
from cv_bridge import CvBridge
import numpy as np
import os
tout_image = os.listdir("image2")


for name_img in tout_image:

    # to actually visualize the effect of `CHAIN_APPROX_SIMPLE`, we need a proper image
    image1 = cv2.imread('image2/' + name_img)
    img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    ret,thresh1 = cv2.threshold(img_gray1, 40, 255, cv2.THRESH_BINARY)
    contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_copy2 = image1.copy()

    cv2.drawContours(image_copy2, contours2, -1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite('image_contour/contour_'+ name_img, image_copy2)

    image_copy3 = image1.copy()
    for i, contour in enumerate(contours2): # loop over one contour area
       for j, contour_point in enumerate(contour): # loop over the points
           #draw a circle on the current contour coordinate
           cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite('image_point_simple/point_simple_' + name_img, image_copy3)
    print('image ' + name_img + ' a été traitée')