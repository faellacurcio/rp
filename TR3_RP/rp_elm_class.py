import numpy


class ELM:
    """
    ELM Class

    functions:
    -fit(x,y)
    -test(x)
    """

    def __init__(self, neurons, a=1):
        """
        neurons     number of hidden neurons
        a           const value of sigmoid funcion
        """
        self.neurons = neurons
        self.a = a
        self.mat_m = numpy.array([])
        self.mat_z = numpy.array([])
        self.mat_w = numpy.array([])

    def add_bias(self, mat):
        """
        Add bias
        """
        return numpy.concatenate((self.a*numpy.array([numpy.ones(mat.shape[1])]), mat), axis=0)

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

        p = x_train.shape[0]
        N = x_train.shape[1]

        # Adds BIAS
        x_train = self.add_bias(x_train)

        # Create random values for W
        self.mat_w = numpy.random.rand(self.neurons, p+1)

        # W.X
        mat_u = numpy.dot(self.mat_w, x_train)

        # Sigmoide function
        self.mat_z = 1/(1+numpy.exp(-self.add_bias(mat_u)))

        self.mat_m = numpy.dot(numpy.dot(y_train, self.mat_z.T), numpy.linalg.pinv(
            numpy.dot(self.mat_z, self.mat_z.T)))

    def predict(self, x_test):
        """
        Used trained data to predict new data
        """
        mat_z = numpy.dot(self.mat_w, self.add_bias(x_test))
        mat_z = 1/(1+numpy.exp(-self.add_bias(mat_z)))
        output = numpy.dot(self.mat_m, mat_z)

        return numpy.argmax(output, axis=0)
