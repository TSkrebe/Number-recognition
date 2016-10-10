import numpy as np
import scipy.io
from scipy.spatial.distance import cdist

from Task1.helpers import confusion_matrix


data = scipy.io.loadmat('../svhn.mat')
train_features = np.array(data['train_features'])
train_classes = np.array(data['train_classes'])[0]

test_features = np.array(data['test_features'])
test_classes = np.array(data['test_classes'])[0]


def find_class(indices):
    classes = train_classes[indices]
    count = np.bincount(classes)
    max_value = max(count)
    all_max = [i for i, x in enumerate(count) if x == max_value]
    for c in classes:
        if c in all_max:
            return c


def knn(k=1, n=100, train_data=train_features, test_data=test_features):
    test_data = test_data[:,:n]
    train_data = train_data[:,:n]
    true_results = []
    for p, feature in enumerate(test_data):
        dist = cdist(train_data, [feature], 'euclidean')
        dist = np.squeeze(dist)
        #k smallest indices
        ind = np.argpartition(dist, k)[:k]
        #sort indices by distance
        ind = ind[np.argsort(dist[ind])]
        calculated_class = find_class(ind)
        true_results.append(calculated_class)

    return np.array(true_results)


def knn_classification_confusion(k=1, n=100):
    con_m = confusion_matrix(test_classes, knn(k, n))
    precision = np.sum(con_m.diagonal())/float(np.sum(con_m))
    print "Precision: {}".format(precision)
    print con_m


#knn_classification_confusion(10)
