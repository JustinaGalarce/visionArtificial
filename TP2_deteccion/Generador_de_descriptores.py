import cv2 as cv
import numpy as np
import pandas as pd
from openpyxl import Workbook
def library():
    shapes = ['Circulo/Circulo1.jpeg',
              'Circulo/Circulo2.jpeg',
              'Circulo/Circulo3.jpeg',
              'Circulo/Circulo4.jpeg',
              'Circulo/Circulo5.jpeg',
              'Circulo/Circulo6.jpeg',
              'Circulo/Circulo7.jpeg',
              'Circulo/Circulo8.jpeg',
              'Circulo/Circulo9.jpeg',
              'Circulo/Circulo10.jpeg',
              'Estrellas/Estrella1.jpeg',
              'Estrellas/Estrella2.jpeg',
              'Estrellas/Estrella3.jpeg',
              'Estrellas/Estrella4.jpeg',
              'Estrellas/Estrella5.jpeg',
              'Estrellas/Estrella6.jpeg',
              'Estrellas/Estrella7.jpeg',
              'Estrellas/Estrella8.jpeg',
              'Estrellas/Estrella9.jpeg',
              'Estrellas/Estrella10.jpeg',
              'Triangulo/Triangulo1.jpeg',
              'Triangulo/Triangulo2.jpeg',
              'Triangulo/Triangulo3.jpeg',
              'Triangulo/Triangulo4.jpeg',
              'Triangulo/Triangulo5.jpeg',
              'Triangulo/Triangulo6.jpeg',
              'Triangulo/Triangulo7.jpeg',
              'Triangulo/Triangulo8.jpeg',
              'Triangulo/Triangulo9.jpeg',
              'Triangulo/Triangulo10.jpeg']
    return shapes

def labels(): # 1 = circulo, 2 = estrella, 3 = triangulo
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    return labels

def to_gray(im):
    return cv.cvtColor(cv.imread(im), cv.COLOR_BGR2GRAY)

def to_binary(imGray):
    _, im = cv.threshold(imGray, 128, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return im

def find_contours(imBinary):
    contours, hierarchy = cv.findContours(imBinary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours

def getBiggerContour(contours):
    max_ctn = contours[0]
    for ctn in contours:
        if cv.contourArea(ctn) > cv.contourArea(max_ctn):
            max_ctn = ctn
    return max_ctn

def final_contoursArray(imBinary):
   return getBiggerContour(find_contours(imBinary))


def hu_moments(contour): #array con imagenes contorneadas (getContours y get biggerContourns)
    return cv.HuMoments(cv.moments(contour))

def get_dataset(array, num):
    dataset = [[i in range(7)]for i in range(num)] #obvio que el error era aca la reputa madre
    for i in range(num):
        grays = to_gray(array[i]) ### mirar
        binaries = to_binary(grays)
        final_contour = final_contoursArray(binaries) # nos devuelve el contorno de cada imagen
        hu = hu_moments(final_contour)
        hu = [hu[0][0],hu[1][0],hu[2][0],hu[3][0],hu[4][0],hu[5][0],hu[6][0]]
        dataset[i] = hu  # nos devuelve el un array con los arrays de huMoments de cada imagen
    return dataset

def main ():
    tags = labels()
    data = get_dataset(library(), len(tags))
    #le hacemos log porque sino el codigo hace pum
    for i in range(30):
        for j in range(7):
            data[i][j] = -1 * np.copysign(1.0, data[i][j]) * np.log10(np.absolute(data[i][j]))
    hu1 = [data[i][0] for i in range(len(tags))]
    hu2 = [data[i][1] for i in range(len(tags))]
    hu3 = [data[i][2] for i in range(len(tags))]
    hu4 = [data[i][3] for i in range(len(tags))]
    hu5 = [data[i][4] for i in range(len(tags))]
    hu6 = [data[i][5] for i in range(len(tags))]
    hu7 = [data[i][6] for i in range(len(tags))]
    dataset = pd.DataFrame({"tags": tags, "hu1": hu1, "hu2": hu2, "hu3": hu3, "hu4": hu4, "hu5": hu5, "hu6": hu6, "hu7": hu7})
    dataset.to_excel('data.xlsx', sheet_name='sheet1', index=False)

main()










