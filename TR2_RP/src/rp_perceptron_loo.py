"""
rp_perceotron_loo.py
Leave one out Perceptron
"""

import matplotlib.pyplot as plt

from plot_confusion_matrix import plot_confusion_matrix
import numpy
from sklearn.model_selection import LeaveOneOut
from rp_perceptron_class import Perceptron
#from sklearn.linear_model import Perceptron

def zscore(x_value):
    """
    zscore function
    """
    x_value = x_value - numpy.mean(x_value, axis=0)
    x_value = x_value / numpy.std(x_value, axis=0, ddof=1)
    return x_value

for epoch_value in range(1,9):
    print("For "+str(epoch_value)+" epochs:")
    samples = list()
    # Abre e lê dermatology como um dataset
    with open('dermatology.dat') as iris:
        for row in iris.readlines():
            samples.append(list(map(float, row.split(","))))
    dataset = numpy.array(samples)

    # separa os dados em features e classe (X/Y)
    X = dataset[:, :-1]

    for classes in list(set(list(dataset[:, -1]))):
        classes = int(classes)
        confusion_y_test = []
        confusion_y_pred = []
        b = list(dataset[:, -1:])
        Y = [int(x == classes) for x in b]
        Y = numpy.array(Y)

        # X = zscore(X)
        clf = Perceptron(max_iter=5, tol=-numpy.infty)
        #clf = Perceptron()

        # Separa as amotras utilizando leave one ou
        cross_val = LeaveOneOut()
        cross_val.get_n_splits(X)

        total = len(X)
        success = 0.0

        # loop que roda a classificação nos diferentes grupos de validação
        for train_index, test_index in cross_val.split(X, Y):

            #Separa em treinamento e amostras
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            # classifica as amostras da base de teste
            clf.fit(X_train, Y_train, epoch_value)
        
            y = clf.predict(X_test)
            y.squeeze()
            Y_test = Y_test[numpy.newaxis]
            Y_test.squeeze()

            confusion_y_test.extend(Y_test[0]+numpy.ones([len(Y_test)]))
            confusion_y_pred.extend(y[0]+numpy.ones([len(y)]))
            #armazena o sucesso
            success += int(y == Y_test)

        # calcula e imprime o resultado.
        result = 100*(success/total)

        confusion_y_test = numpy.array(confusion_y_test).astype(int)
        confusion_y_pred = numpy.array(confusion_y_pred).astype(int)
        #plot_confusion_matrix(confusion_y_test, confusion_y_pred, classes=["class "+str(classes),"Other Classes"],                    title='Perceptron, LOO, class '+str(classes)+', 100 epochs: '+str('%.2f' % result)+'%')
    
        #plt.savefig("q3_fig_loo_classe"+str(classes))
        #plt.clf()
        print('for class:%d Result:%.2f %%' % (classes, result))
