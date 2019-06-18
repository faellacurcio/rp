"""
    ELM Class
"""

import numpy
import random


class RBF:
    """
    ELM Class

    functions:
    -fit(x,y)
    -test(x)
    """

    def __init__(self, neurons, num_neuronios=40):
        """
        neurons     number of hidden neurons
        a           const value of sigmoid funcion
        """
        self.num_neuronios = num_neuronios
        self.centroides = []
        self.mat_m = numpy.array([])
        self.mat_z = numpy.array([])
        self.mat_w = numpy.array([])

    def add_bias(self, mat):
        """
        Add bias
        """
        return numpy.concatenate((numpy.array([numpy.ones(mat.shape[1])]), mat), axis=0)

    def fit(self, x_train, y_train):
        """
        x_train     features vector input
        y_train     class
        p           # of features
        N           # of features Vector
        """
        # Converts to numpy array
        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)

        self.centroides = random.choices(
            numpy.transpose(x_train), k=self.num_neuronios)

        self.mat_z = numpy.ones([1, x_train.shape[1]])

        for element in self.centroides:
            MAT_D = x_train - \
                numpy.tile(numpy.array([element]).T, (1, x_train.shape[1]))
            MAT_D_NORM = numpy.linalg.norm(MAT_D, axis=0)  # ?axis?
            MAT_D_NORM = numpy.array([MAT_D_NORM])
            output = numpy.exp(-MAT_D_NORM**2)
            # output = numpy.exp(-MAT_D_NORM**2)
            self.mat_z = numpy.concatenate((self.mat_z, output), axis=0)

        self.mat_w = numpy.array([numpy.dot(numpy.dot(y_train, self.mat_z.T), numpy.linalg.pinv(
            numpy.dot(self.mat_z, self.mat_z.T)))])

    def predict(self, x_test):
        """
        Used trained data to predict new data
        """
        mat_z_test = numpy.ones([1, x_test.shape[1]])

        for element in self.centroides:
            aux1 = numpy.tile(numpy.array([element]).T, (1, x_test.shape[1]))
            MAT_D_test = x_test - aux1

            MAT_D_NORM_test = numpy.linalg.norm(MAT_D_test, axis=0)  # ?axis?
            MAT_D_NORM_test = numpy.array([MAT_D_NORM_test])
            output = numpy.exp(-MAT_D_NORM_test**2)
            mat_z_test = numpy.concatenate((mat_z_test, output), axis=0)

        return numpy.dot(self.mat_w, mat_z_test)
