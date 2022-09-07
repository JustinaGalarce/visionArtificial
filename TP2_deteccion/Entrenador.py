from sklearn import *
from joblib import dump
import xlwings as xw
from sklearn.model_selection import train_test_split # Import train_test_split function

# dataset
ws = xw.Book("data.xlsx").sheets['sheet1']
y = ws.range("A2:A31").value
hu1 = ws.range("B2:B31").value
hu2 = ws.range("C2:C31").value
hu3 = ws.range("D2:D31").value
hu4 = ws.range("E2:E31").value
hu5 = ws.range("F2:F31").value
hu6 = ws.range("G2:G31").value
hu7 = ws.range("H2:H31").value
x = [
    [float(hu1[i]),
     float(hu2[i]),
     float(hu3[i]),
     float(hu4[i]),
     float(hu5[i]),
     float(hu6[i]),
     float(hu7[i])]
    for i in range(30)
]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

# entrenamiento
clasificador = tree.DecisionTreeClassifier().fit(x_train, y_train)
y_pred=clasificador.predict(x_test)
# visualización del árbol de decisión resultante
_=tree.plot_tree(clasificador)

# guarda el modelo en un archivo
dump(clasificador, 'filename.joblib')



