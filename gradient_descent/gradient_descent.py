import numpy as np
import numpy.random


def batchGradientDescent(computeGradient, computeFunc, epsilon, step,
                         x0: "initilization of x",
                         k_max: "the max iterations"):
    '''this function use batch gradient descent method to optimize the given function'''
    k = 0  #iteration counter
    f_previous = 0
    f_current = computeFunc(x0)
    x = x0
    while True:
        if k > k_max:
            break
        gradient = computeGradient(x)
        x -= step * gradient
        f_previous = f_current
        f_current = computeFunc(x)
        if abs(f_current - f_previous) < epsilon:
            break
        k += 1
    return x, f_current, k


def linearTransformation(theta, X: np.matrix):
    if (X.shape[1] == 1):
        #single vector
        return theta.T * X
    else:
        #vector bundle
        return X * theta

def linearTransformationGradient(theta, X: np.matrix):
    return X.T


def squareCost(theta, X: np.matrix, y, h):
    '''cost = (h(X) - y) ^ 2'''
    return np.sum(np.square(h(theta, X) - y))


def squareCostGradient(theta, X, y, h, h_gradient):
    '''gradient = 2 * h' * (h(X) - y) '''
    return 2 * h_gradient(theta, X) * (h(theta, X) - y)


def union_shuffled_copies(arr1, arr2):
    '''shuffle two array randomly'''
    #assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]


def miniBatchStochasticGradientDescent(X,
                              y,
                              theta0: "init theta",
                              k_max: "max iteration count",
                              step,
                              batchSize,
                              computeFunc=squareCost,
                              computeGradient=squareCostGradient,
                              h=linearTransformation,
                              h_gradient=linearTransformationGradient,
                              epsilon=0.01):
    '''return optimized point `theta`, current function value and the count of iteration\n
    the default function is square function and h is just a linear funtion\n
    the default configuration is the formula: cost(theta) = (X * theta - y) ^ 2'''
    m = len(X)
    k = 0
    theta = theta0
    f_previous = 0
    f_current = computeFunc(theta, X, y, h) / m
    deviations = []  #for test
    while (True):
        if k > k_max:
            break
        batchIndexs = np.random.choice(m, batchSize)
        X_batch = X[batchIndexs]
        y_batch = y[batchIndexs]
        theta -= step * computeGradient(theta, X_batch, y_batch, h, h_gradient)
        '''
        X, y = union_shuffled_copies(X, y)
        for i in range(m):
            theta -= step * computeGradient(theta, X[i].T, y[i,0], h, h_gradient)
        '''
        f_previous = f_current
        f_current = computeFunc(theta, X, y, h) / m
        if abs(f_current - f_previous) < epsilon:
            break
        deviations.append(abs(f_current - f_previous))
    return theta, f_current, k
