import numpy

class ELM:
    def __init__(self, neurons, a=1):
        """
        neurons     number of hidden neurons
        a           const value of sigmoid funcion
        """
        self.neurons = neurons
        self.a = a
        self.mat_M = numpy.array([])
        self.mat_Z = numpy.array([])
        self.mat_W = numpy.array([])
    
    def addBias(self, mat):
        return numpy.concatenate((self.a*numpy.array([numpy.ones(mat.shape[1])]), mat), axis = 0)

    def fit(self, X_train, Y_train):
        """
        X_train     features vector input
        Y_train     class
        p           # of features
        N           # of features Vector
        """
        
        # Converts to numpy array
        X_train = numpy.array(X_train)
        Y_train = numpy.array(Y_train)
        
        p = X_train.shape[0]
        N = X_train.shape[1]

        # Adds BIAS
        X_train = self.addBias(X_train) # numpy.concatenate((self.a*numpy.array([numpy.ones(N)]), X_train), axis = 0)
        
        # Create random values for W
        self.mat_W = numpy.random.rand(self.neurons,p+1)

        # W.X
        mat_U = numpy.dot(self.mat_W,X_train)

        # Sigmoide function
        self.mat_Z = 1/(1+numpy.exp(-1*mat_U))

        #Add bias to output of hidden layer
        self.mat_Z =  self.addBias(self.mat_Z)

        self.mat_M = numpy.dot(numpy.dot(Y_train,self.mat_Z.T),numpy.linalg.inv(numpy.dot(self.mat_Z,self.mat_Z.T)))
    
    def test(self, X_test):
        mat_z = numpy.dot(self.mat_W,self.addBias(X_test))
        mat_z = 1/(1+numpy.exp(-1*mat_z))
        output = numpy.dot(self.mat_M,self.addBias(mat_z))
        return numpy.argmax(output,axis=0)

