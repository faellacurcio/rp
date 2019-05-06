#importa as bibliotecas
import numpy
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier

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

# Função de normalização
def zscore(X):
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X

# X = zscore(X)
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='lbfgs', verbose=0, tol=1e-2, random_state=1,
                    learning_rate_init=.1)

confusion_y_test = []
confusion_y_pred = []

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
    clf.fit(X_train, Y_train)

    y = clf.predict(X_test)

    #armazena o sucesso
    y = numpy.argmax(y,axis=1)
    Y_test = numpy.argmax(Y_test,axis=1)

    confusion_y_test.extend(Y_test)
    confusion_y_pred.extend(y)

    success += sum(y == Y_test)

# calcula e imprime o resultado.
result = 100*(success/total)
print('%.2f %%' % (result))

plot_confusion_matrix(confusion_y_test, confusion_y_pred, classes=["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"],
    title='MLP, 100 neurons, Leave-one-out: '+str('%.2f' % result)+'%')

plt.savefig("q3_mlp_loo_notNorm")
plt.show()
plt.clf()