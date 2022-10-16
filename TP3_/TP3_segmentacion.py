import cv2 as cv
import numpy as np
from matplotlib import pyplot as pl
from skimage import color


print("Pulse ESC para terminar.")
yeast = cv.imread('levadura.png') # yeast = levadura

def to_binary(img, value):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, value, 255, cv.THRESH_BINARY)
    return binary

def apply_colorMap(im):
    dst = cv.applyColorMap(im, cv.COLORMAP_JET)
    return dst

def contours(binary):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours

def lam(x):
    pass

def main():
    global seeds
    cv.namedWindow('Binary')
    cv.createTrackbar('Bi_tb', 'Binary', 0, 255, lam)


    while True:
        tecla = cv.waitKey(30)

        # contador de nucleos
        cv.imshow("original_im", yeast)
        value_b = cv.getTrackbarPos('Bi_tb', 'Binary')
        im_bw = to_binary(yeast, value_b)
        cv.imshow('Binary', im_bw)
        num = len(contours(im_bw))
        print(num)
        colorMap = apply_colorMap(im_bw)
        cv.imshow('Color_map', colorMap)

main()

img = cv.imread('C:/Users/agusv/PycharmProjects/visionArtificial/TP3_/levadura.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

cv.imshow("img", thresh)

cv.waitKey()
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(closing, kernel, iterations=3)

cv.imshow("sure_bg", sure_bg)

cv.waitKey()
# Finding sure foreground area
sure_fg = cv.erode(closing, kernel, iterations=3)

# SE DEBERIA USAR EL DISTANCE TRANSFORM ANTES Q EL ERODE
# PQ LOS OBJETOS ESTAN TODOS PEGADOS PERO DE BAJA PQ NOSE SI HAY Q EXPLCIAR ESO
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

cv.imshow("sure_fg", sure_fg)

cv.waitKey()
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv.connectedComponents(sure_fg)
print(markers)
print("------------------")
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
# img[markers == -1] = [0,0,0]
# im2 = color.label2rgb(markers, bg_label=1)
print(img)
print(markers)

yeast[markers == -1] = [255, 0, 0]
# cv.imshow("im2", im2)

cv.waitKey(0)
cv.imshow("img", img)

cv.waitKey()
value_b = cv.getTrackbarPos('Bi_tb', 'Binary')
im_bw = to_binary(yeast, value_b)
























