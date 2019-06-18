"""
TRABALHO 3 QUESTÃO 1
"""
import numpy
from sklearn.model_selection import LeaveOneOut
from rp_elm_class import ELM


def fix_Y(vec_input):
    mat = numpy.zeros([vec_input.shape[0], int(numpy.amax(vec_input))])
    for idx, element in enumerate(Y):
        mat[idx][int(element)-1] = 1
    return mat


def zscore(X):
    """
    Função de normalização
    """
    X = X - numpy.mean(X, axis=0)
    X = X / numpy.std(X, axis=0, ddof=1)
    return X


def pca_convert(mat_x, main_components):
    """
    mat_x = [[x0 x1 x2 x3] [x0 x1 x2 x3]......... [x0 x1 x2 x3]]
    """
    media = numpy.mean(mat_x, axis=0)

    result = mat_x-media

    cov_mat = numpy.cov(result)

    eigen = numpy.linalg.eig(cov_mat)
    eigen_val = eigen[0]
    eigen_vec = eigen[1]

    new_eigen_vec = numpy.zeros([len(eigen_val), len(eigen_val)])

    for i in range(eigen_val.shape[0]):
        index = numpy.argmax(eigen_val[i])
        for j in range(eigen_vec.shape[1]):
            new_eigen_vec[i, j] = numpy.absolute(eigen_vec[j, index])

    result = mat_x*new_eigen_vec[:, :mat_x.shape[1]]

    return result[:, 0:main_components]


SAMPLES = list()
# Abre e lê iris_log como um DATASET
with open('iris_log.dat') as iris:
    for row in iris.readlines():
        SAMPLES.append(list(map(float, row.split())))
DATASET = numpy.array(SAMPLES)
# separa os dados em features e classe (X/Y)


for neuronios in range(1, 18, 2):
    X = DATASET[:, 0:len(DATASET[0])-3]
    Y = DATASET[:, len(DATASET[0])-3:]
    print("--------------")
    for num_componentes in range(len(X[0])-1, 0, -1):

        # Normaliza os dados
        # X = zscore(X)

        # Diminui a dimensaionalidade usando PCA
        X = pca_convert(X, num_componentes)

        # Separa as amotras utilizando leave one ou
        CROSS_VAL = LeaveOneOut()
        CROSS_VAL.get_n_splits(X)

        TOTAL = len(X)
        SUCCESS = 0.0

        # loop que roda a classificação nos diferentes grupos de validação
        for train_index, test_index in CROSS_VAL.split(X, Y):

            # Separa em treinamento e amostras
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            ELM_OBJ = ELM(neuronios)

            # classifica as amostras da base de teste
            ELM_OBJ.fit(X_train.T, Y_train.T)

            y = ELM_OBJ.predict(X_test.T)

            # armazena o sucesso
            SUCCESS += sum([y] == numpy.argmax(Y_test))

        # calcula e imprime o resultado.
        result = 100*(SUCCESS/TOTAL)
        print("Resultado utilizando "+str(neuronios) +
              " neuronios no ELM e "+str(num_componentes)+" componentes no PCA")
        print('%.2f %%' % (result))
    del num_componentes
