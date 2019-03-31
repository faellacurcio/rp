#importa as bibliotecas
import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

#Função de LDA
def lda(X_train, Y_train, X_test):
    y = list()
    centroids = list()
    covariance = list()
    
    # calcula a matriz de covariancia e multiplica pela matriz identidade
    covariance.append(numpy.cov(X_train.T)*numpy.identity(4))
    
    for class_ in sorted(list(set(Y_train))):
        idx = numpy.where(Y_train == class_)[0]
        # Para cada classe enontra a média das caracteristicas
        centroids.append(numpy.mean(X_train[idx], axis=0))
    
    for x in X_test:
        dist = list()
        # para cada classe aplica a função descrita nos slides para o LDA
        for idx in sorted(list(set(Y_train))):
            dist.append(
                numpy.dot(
                    numpy.dot(
                        (x - centroids[idx]), 
                        numpy.linalg.inv(covariance) 
                    ), 
                    numpy.transpose(x-centroids[idx])
                )
            )
        y_ = numpy.argmin(dist)
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
    y = lda(X_train, Y_train, X_test)

    #armazena o sucesso
    success += sum(y == Y_test)/len(Y_test)

# calcula e imprime o resultado.
result = (100*(success/20))
print('%.2f %%' % (result))