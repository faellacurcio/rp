import numpy as np

from sklearn.decomposition import PCA


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


X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])

pca = PCA(n_components=2)

pca.fit(X)

print(pca.singular_values_)
