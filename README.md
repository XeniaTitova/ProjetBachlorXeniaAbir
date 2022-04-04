# ProjetBachlorXeniaAbir

pour executé le projet il faut l'importé en entier, telecharger au même endroit le fichier mori_aruco_new.bag
puis executé les commandes :


>>> python3 bag2png.py            #extraire les fichier du .bag dans le fichier /image
>>> python3 contour_test1.py      #trouvé le contours de images et les reenregistré (pas nésesaire)
>>> python3 code_complet.py       #detecte les aruco (stock dans  /image_aruco_detect), 
                                  #detecte axe (dans /image_axe_detect), 
                                  #trouve position aproximé des point (dans  image_pts_3Dto2D)
                                  #precise ces point (dans /image_treter)
                                  #remplie le fichier data.txt avec les coordoné des angles 
