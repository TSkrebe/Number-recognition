import numpy as np
import matplotlib.pyplot as plt
from compute_pca import compute_pca


def show_cumulative_variance():

    import scipy.io
    data = scipy.io.loadmat('svhn.mat')
    train_data = np.array(data['train_features'])

    cumsum = np.cumsum(compute_pca(train_data)[1])

    x_axis = [x for x in range(len(cumsum))]
    plt.plot(x_axis, cumsum, 'ro')
    plt.xlabel("Principal components")
    plt.ylabel("Comulative variance")
    plt.show()


show_cumulative_variance()