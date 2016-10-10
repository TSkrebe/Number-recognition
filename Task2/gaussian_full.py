import numpy as np
import scipy.io

from Task1.helpers import confusion_matrix, mean, cov_matrix

data = scipy.io.loadmat('../svhn.mat')

train_features = np.array(data['train_features'])
train_classes = np.array(data['train_classes'])[0]

test_features = np.array(data['test_features'])
test_classes = np.array(data['test_classes'])[0]


def gaussian_full(features=100, train_data=train_features, test_data=test_features):
    train_data = train_data[:,:features]
    test_data = test_data[:,:features]
    means = []
    covariances = []
    for c in range(1, 11):
        indices = [x for x, y in enumerate(train_classes) if c == y]
        class_data = train_data[indices]
        means.append(mean(class_data))
        covariances.append(cov_matrix(class_data))

    print "Determinants:"
    for ind, c in enumerate(covariances):
        print ind+1, np.linalg.det(c)

    predictions = []
    for feature in test_data:
        results = [log_gaussian(feature, m, c) for m, c in zip(means, covariances)]
        predictions.append(np.argmax(results)+1)

    return np.array(predictions)


def log_gaussian(x, m, c):
    return -0.5 * (np.log(np.linalg.det(c)) + np.dot(np.dot((x-m), np.linalg.inv(c)), (x-m)))


def full_gaussian_classification_confusion(n=100):
    con_m = confusion_matrix(test_classes, gaussian_full(n))
    precision = np.sum(con_m.diagonal())/float(np.sum(con_m))
    print "Precision: {}".format(precision)
    print con_m


#full_gaussian_classification_confusion()
