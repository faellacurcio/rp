#importa as bibliotecas
import numpy
from rp_elm_class import ELM
from sklearn.model_selection import StratifiedShuffleSplit

samples = list()
# Abre e lê iris_log como um dataset
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split())))
dataset = numpy.array(samples)

# separa os dados em features e classe (X/Y) 
X = dataset[:,0:len(dataset[0])-3]
Y = dataset[:,len(dataset[0])-3:]
Y2 = numpy.argmax(Y, axis=1)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

X = zscore(X)
obj_ELM = ELM(10)

# Separa as amotras utilizando holdout
cross_val = StratifiedShuffleSplit(n_splits=20, test_size=0.3)
cross_val.get_n_splits(X)

success = 0.0

# loop que roda a classificação nos diferentes grupos de validação
for train_index, test_index in cross_val.split(X,Y2):
    #Separa em treinamento e amostras
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    # classifica as amostras da base de teste
    # classifica as amostras da base de teste
    obj_ELM.fit(X_train.T, Y_train.T)

    y = obj_ELM.test(X_test.T)

    #armazena o sucesso
    Y_test = numpy.argmax(Y_test,axis=1)
    success += sum(y == Y_test)/len(Y_test)

# calcula e imprime o resultado.
result = (100*(success/20))
print('%.2f %%' % (result))