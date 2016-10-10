import numpy as np


def cov_matrix(matrix, subtract=0):
    matrix = matrix - mean(matrix)
    return np.transpose(matrix).dot(matrix)/(matrix.shape[0] - subtract)


def mean(matrix, axis=0):
    return np.sum(matrix, axis=axis)/float(matrix.shape[axis])


def confusion_matrix(actual, calculated):
    confusion = np.zeros((10, 10), dtype=int)
    for a, c in zip(actual, calculated):
        confusion[a-1, c-1] += 1
    return confusion

