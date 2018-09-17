import numpy as np
import numpy.linalg


#Regularization
def L1ParameterRegularization(theta):
    return np.sum(abs(theta))


def L2ParameterRegularization(theta):
    return theta.T * theta


#linear function
def linearTransformation(theta, X: np.matrix):
    if (X.shape[1] == 1):
        #single vector
        return theta.T * X
    else:
        #vector bundle
        return X * theta


def linearTransformationGradient(theta, X: np.matrix):
    return X.T

#MSE cost function
def squareCost(theta, X: np.matrix, y, h):
    '''cost = (h(X) - y) ^ 2'''
    return np.sum(np.square(h(theta, X) - y))


def squareCostGradient(theta, X, y, h, h_gradient):
    '''gradient = 2 * h' * (h(X) - y) '''
    return 2 * h_gradient(theta, X) * (h(theta, X) - y)