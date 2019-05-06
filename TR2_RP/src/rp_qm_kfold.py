#importa as bibliotecas
import numpy
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

class MQ:
    def __init__(self):
        self.mat_M = []

    def activation(self, matrix):
        return 1/(1+np.exp(matrix))

    def addBias(self, matrix):
        a = np.ones([1,matrix.shape[1]])
        matrix = np.concatenate((a, matrix), axis=0)
        return matrix

    def train(self, X_train, Y_train):
        a = np.dot(Y_train,X_train.T)
        b = np.dot(X_train,X_train.T)
        c = numpy.linalg.pinv(b)
        self.mat_M = np.dot(a,c)


    def test(self, X_test):
        return( np.dot(self.mat_M, X_test) )

samples = list()
# Abre e lê iris_log como um dataset
with open('dataset/iris_log.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)

# separa os dados em features e classe (X/Y) 
X = dataset[:,0:len(dataset[0])-3]
Y = dataset[:,len(dataset[0])-3:]
#Y = numpy.argmax(Y, axis=1)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

X = zscore(X)
maxQuadrados = MQ()

#Função de 1-NN
def one_nn(X_train, Y_train, X_test):
    y = list()
    for x in X_test:
        # Calcula a distancia para todos os pontos
        dist = numpy.linalg.norm(X_train - x, axis=1)
        # Seleciona a mínima distancia
        y_ = Y_train[numpy.argmin(dist)]
        # cria vetor de estimativa
        y.append(y_)
    return numpy.array(y)

# Separa as amotras utilizando holdout
cross_val = StratifiedShuffleSplit(n_splits=20, test_size=0.3)
cross_val.get_n_splits(X)

success = 0.0

# loop que roda a classificação nos diferentes grupos de validação
for train_index, test_index in cross_val.split(X,Y):
    #Separa em treinamento e amostras
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    # classifica as amostras da base de teste
    maxQuadrados.train(X_train.T, Y_train.T)

    y = maxQuadrados.test(X_test.T)
    y = np.argmax(y,axis=0)
    Y_test2 = np.argmax(Y_test,axis=1)
    #armazena o sucesso
    success += sum(y == Y_test2)/len(Y_test)

# calcula e imprime o resultado.
result = (100*(success/20))
print('%.2f %%' % (result))