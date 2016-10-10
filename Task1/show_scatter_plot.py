import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

from apply_pca import apply_pca


def show_scatter_plot():

    import scipy.io
    data = scipy.io.loadmat('svhn.mat')
    train_classes = data['train_classes'][0]

    train_features = data['train_features']

    colors = ['yellow', 'violet', 'green', 'oldLace', 'black', 'brown', 'orange', 'blue', 'gray', 'red']
    X_PCA = apply_pca(train_features)
    for i, c in enumerate(colors):
        indices = np.where(train_classes == i+1)
        data = X_PCA[indices]
        plt.scatter(data[:,0], data[:,1], label=(i+1), c=c)

    plt.legend(loc=4)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.show()

show_scatter_plot()