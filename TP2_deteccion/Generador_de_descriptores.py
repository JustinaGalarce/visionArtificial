import cv2 as cv
import numpy as np
def library():
figuras = {"WhatsApp Image 2022-09-02 at 17.01.48.jpeg", "WhatsApp Image 2022-09-02 at 17.01.48.jpeg", "WhatsApp Image 2022-09-02 at 17.01.48.jpeg", "WhatsApp Image 2022-09-02 at 17.01.48.jpeg", "WhatsApp Image 2022-09-02 at 17.01.48.jpeg", "WhatsApp Image 2022-09-02 at 17.01.48.jpeg", "WhatsApp Image 2022-09-02 at 17.01.48.jpeg", }
def to_gray(array):
    grays = np.empty(array.size)
    for i in array:
        grays[i] = cv.cvtColor(array[i], cv.COLOR_BGR2GRAY)
    return grays

def to_binary(array):
    binaries = np.empty(array.size)
    for i in array:
        ret, binaries[i] = cv.threshold(array[i], 128, 255, cv.THRESH_BINARY_INV)
    return binaries

def hu_moments(array):
    moments = np.empty(array.size)
    for i in array:
        moments[i] = cv.moments(array[i])
    huMoments = np.empty(moments.size)
    for i in moments:
        huMoments = cv.HuMoments(moments[i])
    return huMoments

def main():


main()






