from sklearn import tree
from joblib import dump, load
from Generador_de_descriptores import library, labels, to_gray, to_binary, find_contours, getBiggerContour, final_contoursArray, hu_moments

# dataset


X = get_dataset(library(), 30)

Y = labels()

# entrenamiento
clasificador = tree.DecisionTreeClassifier().fit(X, Y)

# visualización del árbol de decisión resultante
tree.plot_tree(clasificador)

# guarda el modelo en un archivo
dump(clasificador, 'filename.joblib')

# en otro programa, se puede cargar el modelo guardado
clasificadorRecuperado = load('filename.joblib')



