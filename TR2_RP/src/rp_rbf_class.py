import numpy

class RBF:
    def __init__(self, centroids, a=1):
        """
        neurons     number of hidden neurons
        centroids   value of the centroid for each class
        a           const value of sigmoid funcion
        """
        self.centroids = centroids
        self.a = a
        self.mat_M = numpy.array([])
        self.mat_Z = numpy.array([])
        self.mat_W = numpy.array([])
        


    def addBias(self, mat):
        """
        X_train     features vector input
        Y_train     class
        p           # of features
        N           # of features Vector
        """
        return numpy.concatenate((self.a*numpy.array([numpy.ones(mat.shape[1])]), mat), axis = 0)

        
    def fit(self, X_train, Y_train):
        # Converts to numpy array
        X_train = numpy.array(X_train)
        Y_train = numpy.array(Y_train)

        list_X = []
        for line in self.centroids:
            a = (X_train.T-X_train.shape[1]*[line]).T
            b = numpy.linalg.norm(a,axis=0)
            list_X.append(numpy.exp(-1*b**2))

        X_train = numpy.array(list_X)

        p = X_train.shape[0]
        N = X_train.shape[1]

        # Adds BIAS
        X_train = self.addBias(X_train) # numpy.concatenate((self.a*numpy.array([numpy.ones(N)]), X_train), axis = 0)
        
        #Add bias to output of hidden layer
        self.mat_Z =  X_train

        self.mat_M = numpy.dot(numpy.dot(Y_train,self.mat_Z.T),numpy.linalg.pinv(numpy.dot(self.mat_Z,self.mat_Z.T)))

    
    def test(self, X_test):
        list_X = []
        for line in self.centroids:
            a = (X_test.T-X_test.shape[1]*[line]).T
            b = numpy.linalg.norm(a,axis=0)
            list_X.append(numpy.exp(-1*b**2))
        
        X_test = numpy.array(list_X)

        output = numpy.dot(self.mat_M,self.addBias(X_test))
        return numpy.argmax(output,axis=0)

