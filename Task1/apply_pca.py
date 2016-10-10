from compute_pca import compute_pca


def apply_pca(matrix):

    [EVecs, EVal] = compute_pca(matrix)

    E2 = EVecs[:,:2]

    X_PCA = matrix.dot(E2)

    print "2 eigenvalues:", EVal[:2]
    print "5 rows of E2:\n", E2[:5,:]
    print "5 rows of X_PCA:\n", X_PCA[:5,:]

    return X_PCA


# import scipy.io
#
# data = scipy.io.loadmat('svhn.mat')
# X = np.array(data['train_features'])
