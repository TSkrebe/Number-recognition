from Task2.gaussian_full import full_gaussian_classification_confusion
from Task2.gaussian_lda import lda_gaussian_classification_confusion
from Task2.knn import knn_classification_confusion


def classify_with2d():
    print "KNN:"
    knn_classification_confusion(10, 2)
    print "Full Gaussian:"
    full_gaussian_classification_confusion(2)
    print "LDA Gaussian:"
    lda_gaussian_classification_confusion(2)


classify_with2d()