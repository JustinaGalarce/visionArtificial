import cv2 as cv
# Instrucciones en consola
print("Pulse ESC para terminar.")

webcam = cv.VideoCapture(0)


def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius, radius))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing

while True:
    ret, imWebcam = webcam.read()
    cv.imshow('webcam', imWebcam)

    # Aquí se escribe el código para procesar la imagen imWebcam
    im_gray = cv.cvtColor(imWebcam, cv.COLOR_BGR2GRAY)

    # Aquí se escribe el código de visualización
    cv.imshow('blancoYNegro', im_gray)


    #Aca esta el codigo que pasa la imagen de escala de grises a binario
    (thresh, im_bw) = cv.threshold(im_gray, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    #cv.imwrite('bw_image.png', im_bw)
    cv.imshow('Binario', im_bw)

    #Sacamos el ruido con metodo Denoise
    denoise_im_bw = denoise(im_bw, cv.MORPH_ELLIPSE, 3)           #cv.MORPH_RECT metodo elegido.(nose porque)
    cv.imshow('Denoise_im_bw', denoise_im_bw)

    # ESC == 27 en ASCII
    tecla = cv.waitKey(30)
    if tecla == 27:
        break


cv.destroyAllWindows()
