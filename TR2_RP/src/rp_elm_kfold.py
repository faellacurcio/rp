#importa as bibliotecas
import numpy
from rp_elm_class import ELM
from sklearn.model_selection import StratifiedKFold

from plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt

def fix_Y(vec_input):
    mat = numpy.zeros([vec_input.shape[0], int(numpy.amax(vec_input))])
    for idx, element in enumerate(Y):
        mat[idx][int(element)-1] = 1
    return mat


samples = list()
# Abre e lê dermatology como um dataset
with open('dermatology.dat') as iris:
    for row in iris.readlines():
        samples.append(list(map(float, row.split(","))))
dataset = numpy.array(samples)

# separa os dados em features e classe (X/Y) 
X = dataset[:,:-1]
Y = dataset[:,dataset.shape[1]-1]
Y = fix_Y(Y)

Y2 = numpy.argmax(Y, axis=1)

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

X = zscore(X)
obj_ELM = ELM(100)

confusion_y_test = []
confusion_y_pred = []

# Separa as amotras utilizando kfold
cross_val = StratifiedKFold(10)
cross_val.get_n_splits(X)

# loop que roda a classificação nos diferentes grupos de validação
total = len(X)
success = 0.0

for train_index, test_index in cross_val.split(X,Y2):

    #Separa em treinamento e amostras
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # classifica as amostras da base de teste
    obj_ELM.fit(X_train.T, Y_train.T)

    y = obj_ELM.test(X_test.T)
    
    Y_test = numpy.argmax(Y_test,axis=1)

    confusion_y_test.extend(Y_test)
    confusion_y_pred.extend(y)

    #armazena o sucesso
    success += sum(y == Y_test)

# calcula e imprime o resultado.
result = 100*(success/total)
print('%.2f %%' % (result))

plot_confusion_matrix(confusion_y_test, confusion_y_pred, classes=["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"],
    title='ELM, 100 neurons, K-fold: '+str('%.2f' % result)+'%')

plt.savefig("q3_elm_kfold_norm")
plt.show()
plt.clf()