import numpy as np
import scipy.io

from Task1.helpers import mean, cov_matrix, confusion_matrix

data = scipy.io.loadmat('../svhn.mat')

train_features = np.array(data['train_features'])
train_classes = np.array(data['train_classes'])[0]

test_features = np.array(data['test_features'])
test_classes = np.array(data['test_classes'])[0]


def gaussian_lda(features=100, train_data=train_features, test_data=test_features):
    train_data = train_data[:,:features]
    test_data = test_data[:,:features]
    means = []
    covariance = np.zeros([features, features])
    for c in range(1, 11):
        indices = [x for x, y in enumerate(train_classes) if c == y]
        class_data = train_data[indices]
        means.append(mean(class_data))
        covariance += cov_matrix(class_data)
    covariance /= 10
    print "Determinant: {}".format(np.linalg.det(covariance))

    predictions = []
    for feature in test_data:
        results = [discriminant(feature, m, covariance) for m in means]
        predictions.append(np.argmax(results)+1)

    return np.array(predictions)


def discriminant(x, m, c):
    product = np.dot(m, np.linalg.inv(c))
    return np.dot(product, x) - 0.5 * np.dot(product, np.transpose(m))


def lda_gaussian_classification_confusion(n=100):
    con_m = confusion_matrix(test_classes, gaussian_lda(n))
    precision = np.sum(con_m.diagonal())/float(np.sum(con_m))
    print "Precision: {}".format(precision)
    print con_m

#lda_gaussian_classification_confusion()

