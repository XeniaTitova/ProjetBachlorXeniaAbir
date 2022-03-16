from PIL.Image import *
import numpy as np

ii = open('color.png')
(largeur, hauteur)= ii.size

print(range(largeur))
print(range(hauteur))
x = 0
y = 0

while(x<largeur):
    while(y<hauteur):
        (rouge,vert,bleu) = (1,100,200)#ii.getpixel((x,y))
        ii.putpixel((x,y),(bleu,rouge,vert))
        y += 1
    y = 0
    x += 1
 #for x in range(largeur):
#	 for y in range(hauteur):
	#	(rouge,vert,bleu) = ii.getpixel((x,y)) 
 #       ii.putpixel((x,y),(bleu,rouge,vert))

#D:/EPFL18.09.24/EL3_21_22/_projet_de_semestre/color/