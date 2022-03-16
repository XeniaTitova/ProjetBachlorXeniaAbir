import subprocess
import yaml
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
#from PIL.Image import *
 
FILENAME = 'mori_aruco'#nom du fichier .bag
#ROOT_DIR = '/mnt/d/EPFL18.09.24/EL3_21_22/_projet_de_semestre'#emplacement du fichier .bag et ou seront les image (le /mnt/ est nessesaire pour ubuntu sur Windows il est suivie par la lettre du disque)
#BAGFILE = ROOT_DIR + '/' + FILENAME + '.bag'
BAGFILE = FILENAME + '.bag'
 
 
if __name__ == '__main__':
    bag = rosbag.Bag(BAGFILE)
    for i in range(2):
        if (i == 0):
            TOPIC = '/camera/depth/image_rect_raw' #pas besoin de modifier
            DESCRIPTION = 'depth_'
        else:
            TOPIC = '/camera/color/image_raw'   #pas besoin de modifier
            DESCRIPTION = 'color_'
        image_topic = bag.read_messages(TOPIC)
        for k, b in enumerate(image_topic):
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
            cv_image.astype(np.uint8)
            if (DESCRIPTION == 'depth_'):
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
               # cv2.imwrite(ROOT_DIR + '/depth/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
               #cv2.imwrite('depth/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            else:
                cv_image = cv_image[:,:,::-1] #changement de couleur
               # cv2.imwrite(ROOT_DIR + '/image/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)# emplacement ou les images sont enregistrée (ne pas oublier de créé le fichier color)
                cv2.imwrite('image/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            print('saved: ' + DESCRIPTION + str(b.timestamp) + '.png')


    bag.close()

    print('PROCESS COMPLETE')