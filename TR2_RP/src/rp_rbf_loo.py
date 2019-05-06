#importa as bibliotecas
import numpy
from rp_rbf_class import RBF
from sklearn.model_selection import LeaveOneOut
from sklearn.cluster import KMeans

samples = list()
# Abre e lê iris_log como um dataset
with open('iris_log.dat') as iris:
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

# X = zscore(X)
a = KMeans(n_clusters=10, random_state=0).fit(X)
a = a.cluster_centers_

obj_ELM = RBF( a ) 
                #numpy.random.rand(10, X.shape[1]))

# Separa as amotras utilizando leave one ou
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
    obj_ELM.fit(X_train.T, Y_train.T)

    y = obj_ELM.test(X_test.T)

    #armazena o sucesso
    a = numpy.argmax(Y_test,axis=1)
    success += sum(y == a)

# calcula e imprime o resultado.
result = 100*(success/total)
print('%.2f %%' % (result))