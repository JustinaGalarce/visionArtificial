import cv2 as cv
import numpy as np
from matplotlib import pyplot as pl
from skimage import color

print("Pulse ESC para terminar.")
yeast = cv.imread('levadura.png') # yeast = levadura

gray = cv.cvtColor(yeast, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
cv.imshow("original_im", yeast)
cv.waitKey()
contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.imshow("binary", binary)
cv.waitKey()
#contador de contornos
num = len(contours)
print(num) # falta que ande esto (no me cuenta los nucleos)
#watershed

_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

cv.imshow("yeast", thresh)

cv.waitKey()
# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(closing, kernel, iterations=3)

cv.imshow("sure_bg", sure_bg)

cv.waitKey()
# Finding sure foreground area
sure_fg = cv.erode(closing, kernel, iterations=3)
x = cv.dilate(sure_fg)
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)


cv.imshow("sure_fg", x)

cv.waitKey()
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, x)

# Marker labelling
_, markers = cv.connectedComponents(x)
print(markers)
print("------------------")
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(yeast, markers)
yeast[markers == -1] = [0,0,0]
im2 = color.label2rgb(markers, bg_label=1)

print(yeast)
print(markers)

yeast[markers == -1] = [255, 0, 0]
cv.imshow("im2", im2)
cv.imshow("img", yeast)

cv.waitKey()



