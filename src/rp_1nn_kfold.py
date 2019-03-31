#importa as bibliotecas
import numpy
from sklearn.model_selection import StratifiedKFold

samples = list()
# Abre e lê iris_log como um dataset
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)
# separa os dados em features e classe (X/Y) 
X = dataset[:,0:len(dataset[0])-3]
Y = dataset[:,len(dataset[0])-3:]
Y = numpy.argmax(Y, axis=1)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

X = zscore(X)

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

# Separa as amotras utilizando kfold
cross_val = StratifiedKFold(10)
cross_val.get_n_splits(X)

# loop que roda a classificação nos diferentes grupos de validação
total = len(X)
success = 0.0

for train_index, test_index in cross_val.split(X,Y):

    #Separa em treinamento e amostras
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # classifica as amostras da base de teste
    y = one_nn(X_train, Y_train, X_test)

    #armazena o sucesso
    success += sum(y == Y_test)

# calcula e imprime o resultado.
result = 100*(success/total)
print('%.2f %%' % (result))