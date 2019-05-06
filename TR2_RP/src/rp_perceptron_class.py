"""
rp_perceotron_class.py
Perceptron class
"""
import numpy
from numpy import heaviside

class Perceptron:
    """
    Perceptron class
    """
    def __init__(self, learning_rate=0.25, value_a=1, max_iter=0, tol=0):
        """
        neurons     number of hidden neurons
        a           const value of sigmoid funcion
        """
        self.learning_rate = learning_rate
        self.weights = numpy.array([])
        self.value_a = value_a


    def add_bias(self, mat):
        """
        Add_bias function
        """
        return numpy.concatenate(
            (self.value_a*numpy.array([numpy.ones(mat.shape[1])]), mat),
            axis=0
        )


    def prediction(self, x_vector):
        """
        Prediction function
        """
        aux_response = numpy.dot(self.weights, self.add_bias(x_vector.T))
        response =  heaviside(aux_response, 0)
        # response = numpy.sign(aux_response)
        return response


    def fit(self, x_train, y_train, epochs = 5):
        """
        train_weights function
        """
        if self.weights.size == 0:
            self.weights = numpy.zeros([1, x_train.shape[1]+1])

        for _ in range(epochs):
            aux_prediction = self.prediction(x_train)
            aux2_prediction = y_train - aux_prediction
            aux3_prediction = numpy.dot(aux2_prediction, self.add_bias(x_train.T).T)

            self.weights = self.weights + (self.learning_rate*aux3_prediction)


    def predict(self, x_test):
        """
        test function
        """
        aux_prediction = self.prediction(x_test)
        return aux_prediction
