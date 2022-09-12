from joblib import load
import cv2 as cv
import numpy as np

print("Pulse ESC para terminar.")
webcam = cv.VideoCapture(0)

def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius+1, radius+1))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing

def contours(binary, img):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv.contourArea(i)
        if area > 1000:
            cv.drawContours(img, i, -1, (255, 0, 255), 7)
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
    return contours

def to_binary(img, value):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, value, 255, cv.THRESH_BINARY_INV)
    return binary

def find_contours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours

def getBiggerContour(contours):
    max_ctn = contours[0]
    for ctn in contours:
        if cv.contourArea(ctn) > cv.contourArea(max_ctn):
            max_ctn = ctn
            return max_ctn

def setBinaryAutom(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 0, 255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return thresh1

def lam(x):
    pass

def main():
    cv.namedWindow('Binary')
    cv.createTrackbar('Bi_tb', 'Binary', 0, 255, lam)
    cv.namedWindow('Denoise')
    cv.createTrackbar('Denoise_tb', 'Denoise', 0, 5, lam)
    cv.namedWindow('Webcam')
    clasificador = load('filename.joblib')
    while True:
        tecla = cv.waitKey(30)
        ret, imWebcam = webcam.read()

        value_b = cv.getTrackbarPos('Bi_tb', 'Binary')
        im_bw = to_binary(imWebcam, value_b)
        cv.imshow('Binary', im_bw)

        value_d = cv.getTrackbarPos('Denoise_tb', 'Denoise')
        denoise_im_bw = denoise(im_bw, cv.MORPH_ELLIPSE, value_d)
        cv.imshow('Denoise', denoise_im_bw)

        contours1 = contours(denoise_im_bw, imWebcam)
        for i in contours1:
            if cv.contourArea(i) > 1000:
                hu = cv.HuMoments(cv.moments(i))
                huMoments = [hu[i][0] for i in range(7)]
                for j in range(7):
                    huMoments[j] = -1 * np.copysign(1.0, huMoments[j]) * np.log10(np.absolute(huMoments[j]))
                result = clasificador.predict([huMoments])

                x, y, w, h = cv.boundingRect(i)
                cv.rectangle(imWebcam, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if result == 1:
                    cv.putText(imWebcam, "Circulo", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if result == 2:
                    cv.putText(imWebcam, "Estrella", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if result == 3:
                    cv.putText(imWebcam, "Triangulo", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.imshow('Webcam', imWebcam)

        if tecla == 27:
            break

cv.destroyAllWindows()

main()





