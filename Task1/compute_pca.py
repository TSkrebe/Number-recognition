import numpy as np

from Task1.helpers import cov_matrix


def compute_pca(matrix):

    matrix = cov_matrix(matrix, 1)
    (values, vectors) = np.linalg.eig(matrix)
    for ind, head in enumerate(vectors[0]):
        if head < 0:
            vectors[:, ind] = np.negative(vectors[:, ind])
    return vectors, values


# import scipy.io
#
# data = scipy.io.loadmat('svhn.mat')
# matrix = np.array(data['train_features'])

