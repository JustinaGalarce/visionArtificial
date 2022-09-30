import cv2 as cv
import numpy as np

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
        cv.imshow("original_im", yeast)
        value_b = cv.getTrackbarPos('Bi_tb', 'Binary')
        im_bw = to_binary(yeast, value_b)
        cv.imshow('Binary', im_bw)
        #denoise_im = denoise(value_b)
        colorMap = apply_colorMap(im_bw)
        cv.imshow('Color_map', colorMap)
        num = len(contours(im_bw, yeast))
        print(num)



        if tecla == 27:
            break



main()












