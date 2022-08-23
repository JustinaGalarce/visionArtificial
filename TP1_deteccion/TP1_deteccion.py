from PIL import Image
import cv2 as cv
import numpy as np

# Instrucciones en consola
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
    cv.imshow("Contours", img)


def to_binary(img, value):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, value, 255, cv.THRESH_BINARY_INV)
    return binary

def find_contours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours

def get_contours(binary, img):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv.contourArea(i)
        if area > 0.5:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02*peri, True)
    return contours
def getBiggerContour(contours):
    max_ctn = contours[0]
    for ctn in contours:
        if cv.contourArea(ctn) > cv.contourArea(max_ctn):
            max_ctn = ctn
            return max_ctn

def setBinaryAutom(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 0, 255,
                                 cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return thresh1
def library():
    circulo = setBinaryAutom(np.array(Image.open('Circulo.jpeg')))
    triangulo = setBinaryAutom(np.array(Image.open('Triangulo.png')))
    rectangulo = setBinaryAutom(np.array(Image.open('Rectangulo.jpeg')))

    con_rec = find_contours(circulo)
    con_cir = find_contours(triangulo)
    con_tri = find_contours(rectangulo)

    figures = {
        "circulo": getBiggerContour(con_cir),
        "rectangulo": getBiggerContour(con_rec),
        "triangulo": getBiggerContour(con_tri)
    }
    return figures
def match(contour, val):
    contours = library()
    for i in contours.keys():
        distance = cv.matchShapes(contour, contours[i], cv.CONTOURS_MATCH_I2, 0)
        if distance < val:  # el error ponerlo con la barra al tope.
            return i
    return "False"


def main():
    cv.namedWindow('Binary')
    cv.createTrackbar('Bi_tb', 'Binary', 0, 255, to_binary)
    cv.namedWindow('Denoise')
    cv.createTrackbar('Denoise_tb', 'Denoise', 0, 5, denoise)
    cv.namedWindow('Webcam')
    cv.createTrackbar('Error', 'Webcam', 0, 50, denoise)

    while True:
        tecla = cv.waitKey(30)
        ret, imWebcam = webcam.read()

        value_b = cv.getTrackbarPos('Bi_tb', 'Binary')
        im_bw = to_binary(imWebcam, value_b)
        cv.imshow('Binary', im_bw)

        # Sacamos el ruido con metodo Denoise
        value_d = cv.getTrackbarPos('Denoise_tb', 'Denoise')
        denoise_im_bw = denoise(im_bw, cv.MORPH_ELLIPSE, value_d)
        cv.imshow('Denoise', denoise_im_bw)

        # muestra webcam con contornos dibujados
        contours(denoise_im_bw, imWebcam)

        # comparo contornos
        contours1 = get_contours(denoise_im_bw, imWebcam)
        valError = cv.getTrackbarPos('Error', 'Webcam')
        for i in contours1:
            if cv.contourArea(i) > 1000:
                result = match(i, 0.01 + valError / 100)
                if result != "False":
                    x, y, w, h = cv.boundingRect(i)
                    cv.rectangle(imWebcam, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(imWebcam, str(result), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    x, y, w, h = cv.boundingRect(i)
                    cv.rectangle(imWebcam, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow('Webcam', imWebcam)
        # ESC == 27 en ASCII

        if tecla == 27:
            break


cv.destroyAllWindows()

main()