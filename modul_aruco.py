import cv2 as cv
import numpy as np
import subprocess
import yaml
import rosbag
from cv_bridge import CvBridge

def detect_aruco(nom_im = "image_exemple.png", nom_out = " ") :

    if(nom_out == " "):
        nom_out = "marker_detect_" + nom_im
    image = cv.imread(nom_im)

    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_100,)
    #Dictionary dict = aruco::Dictionary::load("ARUCO_MIP_36h12");

    #dictionary = cv.aruco.MarkerMap.setDictionary("ARUCO_MIP_36h12")

    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(image, dictionary, parameters=parameters)


    outputImage = image
    cv.aruco.drawDetectedMarkers(outputImage, markerCorners, markerIds)

    cv.imwrite(nom_out, outputImage)

    print( "fin de detection pour l'image ", nom_im)

def find_pos_3d_to_2d(nom_im = "image_exemple.png", nom_out_aruco = " ", nom_out_axe = " ", 
    nom_out_detect = " ", obj_bleu = [[2,5,0],[1,3,0]] , obj_vert = [[5,5,0],[0,0,0]]):

    if(nom_out_aruco == " "):
        nom_out_aruco = "marker_detect_" + nom_im

    if(nom_out_axe == " "):
        nom_out_axe = "axis_detect_" + nom_im

    if(nom_out_detect == " "):
        nom_out_detect = "points3D_2_2D" + nom_im


    image = cv.imread(nom_im)

    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(image, dictionary, parameters=parameters)

    #print(markerIds)
    a = np.where(markerIds == 1)
    if(len(a[0])) :
        array_pos_origin = a[0][0]
    
        coordinates_origin = markerCorners[array_pos_origin][0][0]
        #print("CAS 1 : Les coordonnées du marker 0 sont : %, %", coordinates_origin)

        outputImage = image.copy()
        cv.aruco.drawDetectedMarkers(outputImage, markerCorners, markerIds)
        markerCorners.append(markerCorners[array_pos_origin])

        cv.imwrite(nom_out_aruco, outputImage)

        #print( "fin de detection pour l'image ", nom_im)

        ###  debut detection axes  ###
        #camera_matrix : mtx // distortion_coefficients0 : dist // translation_vectors : tvecs // rotation_vectors : rvecs

        mtx_camera = np.array([[886.773967, 0.000000, 627.460303], 
            [0.000000, 888.130640, 367.062129],
            [0.000000, 0.000000, 1.000000]])

        distortion = np.array([0.095401, -0.178426, -0.001307, -0.010675, 0.0])

        size_of_marker =  0.029 # side lenght of the marker in meter
        rvecs,tvecs, trash = cv.aruco.estimatePoseSingleMarkers(markerCorners, size_of_marker, cameraMatrix=mtx_camera, distCoeffs=distortion)

        length_of_axis = 0.02
        for i in range(len(tvecs)):
            cv.aruco.drawAxis(outputImage, mtx_camera, distortion, rvecs[i], tvecs[i], length_of_axis)

        #print("enregistrement de l'image avec axe terminée")
        cv.imwrite(nom_out_axe , outputImage)

        ###  debut projection 3D -> 2D  ###
        offset = np.float32([-5.15, 1.45, 0]) ###  hauteur robot 1.2, entre aruco 37mm coté 29
        obj_bleu = np.float32(obj_bleu)
        obj_vert = np.float32(obj_vert)

        zero = np.float32([[0,0,0]]).reshape(-1,3)
        objpts2 = add_offset(obj_bleu, offset).reshape(-1,3) / 100
        objpts = add_offset(obj_vert, offset).reshape(-1,3) / 100

        #objpts = np.float32([[-0.0515,0.1145,0.012], [-0.3515,0.0145,0.012], [-0.2015,0.2645,0.012]]).reshape(-1,3)
        #objpts2 = np.float32([[-4.9,21.15,1.3], [-0.15,12.95,1.3], [-9.55,12.95,1.3], [-33.65,-3.55,1.3],
        #    [-33.65,5.85,1.3], [-25.45,1.2,1.3], [-25.15,24.95,1.3], [-15.75,24.95,1.3], [-20.4,16.75,1.3]]).reshape(-1,3) / 100


        image_copy3 = image.copy()
        
        points2d_vert = dessin_pts_projete(objpts, image_copy3, rvecs, tvecs, mtx_camera, distortion, (0, 255, 0))
        points2d_bleu = dessin_pts_projete(objpts2, image_copy3, rvecs, tvecs, mtx_camera, distortion, (255, 255, 0))
        dessin_pts_projete(zero, image_copy3, rvecs, tvecs, mtx_camera, distortion, (0, 0, 255))

        #print( "fin de projection pour l'image ", nom_im)
        cv.imwrite(nom_out_detect, image_copy3)

        return (points2d_bleu, points2d_vert)
    else : 
        print( "l'image ", nom_im, "n'a pas pu etre treter")
        
        return ([], [])

def dessin_pts_projete(pts, image_copy3, rvecs, tvecs, mtx_camera, distortion, color):
    for i in range(len(rvecs)):
        points2d, _ = cv.projectPoints(pts, rvecs[i], tvecs[i], mtx_camera, distortion)

    for j, point in enumerate(points2d): # loop over the points
    #draw a circle on the current contour coordinate
        cv.circle(image_copy3, (int(point[0][0]), int(point[0][1])), 2, color, 2, cv.LINE_AA)
    
    #pour enlever les parnetese en trop
    A = points2d.astype(int)
    AA = np.zeros((len(A),2))
    for i in range(len(A)):
        #AA.append(A[i][0])
        AA[i][0] = A[i][0][0]
        AA[i][1] = A[i][0][1]

    return AA

def add_offset(A, v):

    for i in range(len(A)):
        A[i] = A[i] + v
    
    return(A)

def trouve_contour(xy, name_img = "image_exemple.png", name_out = " ", h = 25, w = 25, aire_min = 16,  aire_max = 30):

    if(name_out == " "):
        name_out = "avec_point_" + name_img


    img = cv.imread(name_img, cv.IMREAD_COLOR)
        
    height, width = img.shape[:2]

    xy_reel_tot = []
    warning = 0
    impressision = 0
    for coord in xy:
        x = int(coord[0])
        y = int(coord[1])
        
        #print(x,", ",y)
        
        bas = y-h if(y-h > 0) else 0   
        haut = y+h if(y+h < height) else height

        gauche = x-w if(x-w > 0) else 0
        droite = x+w if(x+w < width) else width

        cropped = img.copy()[bas:haut, gauche:droite]

        #cv.imwrite("cropped_" + str(x) + "_" + str(y) + ".png", cropped)

        img_gray1 = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        ret, thresh = cv.threshold(img_gray1, 75, 255, cv.THRESH_BINARY) #2eme paramètre a retravailler pour ajuster contraste
        #thresh = cv.adaptiveThreshold(img_gray1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

        #cv.imwrite("aaa_" + str(x) + "_" + str(y) + "thresh.png", thresh)

        contours,h1 = cv.findContours(thresh,1,2)
        xy_reel = []
        for cnt in contours:
            (a,b),radius = cv.minEnclosingCircle(cnt)

            A = cv.contourArea(cnt)
            if(A<aire_max and A>aire_min and radius<8 and radius>1) :
                approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
                cv.drawContours(cropped,[cnt],0,255,-1)
                M = cv.moments(cnt)
                
                if(M['m00']>0) :
                    cx = int(M['m10']/M['m00']) + gauche
                    cy = int(M['m01']/M['m00']) + bas       #coordoné de l'image entiere   
                    xy_reel.append([cx, cy])
                    #xy_reel_tot.append([cx, cy])
                #print("aire = ", A)
                #print(radius)
            
        if (len(xy_reel) == 1) :    #ok
            xy_reel_tot.append(xy_reel[0])
        else: 
            #xy_reel_tot.append([x, y])
            if (len(xy_reel) == 0) :    #aucun point trouvé (posibilité d'erreur d'axe)
                warning += 1
                xy_reel_tot. append([x, y]) #prendre point aproximé
            else :                          #plusieur points trouvé on prend le plus proche de la valeur aproximé
                impressision += 1
                dist_min = 1000
                for c in xy_reel :
                    dist = np.square(x-c[0]) + np.square(y-c[1])
                    if(dist<dist_min):
                        dist_min = dist
                        cx = c[0]
                        cy = c[1]
                xy_reel_tot.append([cx, cy])
        cv.circle(img, (xy_reel_tot[-1][0], xy_reel_tot[-1][1]), 0, (0, 255, 0), 2) 
        #cv.imwrite("cercle_detect_" + str(x) + "_" + str(y) + ".png", cropped)    
    cv.imwrite(name_out, img)
    W = True if(warning>=3) else False
    I = True if(impressision>=3 or warning) else False

    return(xy_reel_tot,W,I)  
    #print(name_img, "a été traiter, les coordonés éxacte des borts des robots sont:")
    #print(xy_reel)

def bag2png(bagFile, name_out = " ", place = 'image/'):
    '''
    FILENAME = 'mori_aruco_new'#nom du fichier .bag
    #ROOT_DIR = '/mnt/d/EPFL18.09.24/EL3_21_22/_projet_de_semestre'#emplacement du fichier .bag et ou seront les image (le /mnt/ est nessesaire pour ubuntu sur Windows il est suivie par la lettre du disque)
    #BAGFILE = ROOT_DIR + '/' + FILENAME + '.bag'
    BAGFILE = FILENAME + '.bag'
    '''
    if(name_out == " "):
        name_out = "color_"

    if __name__ == '__main__':
        bag = rosbag.Bag(bagFile)
        num_img = 0
        for i in range(2):
            if (i == 0):
                TOPIC = '/camera/depth/image_rect_raw' #pas besoin de modifier
                DESCRIPTION = 'depth_'
            else:
                TOPIC = '/camera/color/image_raw'   #pas besoin de modifier
                DESCRIPTION = name_out
            image_topic = bag.read_messages(TOPIC)
            for k, b in enumerate(image_topic):
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
                cv_image.astype(np.uint8)
                if (DESCRIPTION == 'depth_'):
                    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(cv_image, alpha=0.03), cv.COLORMAP_JET)
                else:
                    cv_image = cv_image[:,:,::-1] #changement de couleur
                    cv.imwrite(place + DESCRIPTION + num_img + '.png', cv_image)
                print('saved: ' + DESCRIPTION + num_img + '.png')
                num_img += 1 

        bag.close()

        print('PROCESS COMPLETE')