#importa as bibliotecas
import numpy
from sklearn.model_selection import LeaveOneOut

samples = list()
# Abre e lê iris_log como um dataset
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
# separa os dados em features e classe (X/Y) 
dataset = numpy.array(samples)
X = dataset[:,0:len(dataset[0])-3]
Y = dataset[:,len(dataset[0])-3:]
Y = numpy.argmax(Y, axis=1)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

X = zscore(X)

#Função de centróide mais proximo
def pmp(X_train, Y_train, X_test):
    y = list()
    centroids = list()
    # para cada classe encontra as amoastras pertencentes
    for class_ in sorted(list(set(Y_train))):
        idx = numpy.where(Y_train == class_)[0]
        # calcula o vetor média contendo as caracteristicas
        centroids.append(numpy.mean(X_train[idx], axis=0))
    for x in X_test:
        # Seleciona a mínima distancia
        dist = numpy.linalg.norm(centroids - x, axis=1)
        y_ = numpy.argmin(dist)
        # cria vetor de estimativa
        y.append(y_)
    return numpy.array(y)

# Separa as amotras utilizando Leave one out
cross_val = LeaveOneOut()
cross_val.get_n_splits(X)

total = len(X)
success = 0.0

# loop que roda a classificação nos diferentes grupos de validação
for train_index, test_index in cross_val.split(X,Y):

    #Separa em treinamento e amostras
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # classifica as amostras da base de teste
    y = pmp(X_train, Y_train, X_test)

    #armazena o sucesso
    success += sum(y == Y_test)

# calcula e imprime o resultado.
result = 100*(success/total)
print('%.2f %%' % (result))