import numpy as np


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


'''
def linearTransformation(X, theta):
    return np.dot(X, theta)


def singleLinearTransformation(x, theta):
    return np.dot(theta.T, x)


def linearMatrix(X, theta):
    return theta


def computeCostFunc(X: "dataset", y: "label", theta: "parameter", h: "function, h(X, theta)"):
    return 0.5 * np.sum((h(X, theta) - y.T)** 2) / len(X)

def computeGradient(X: "dataset", y: "label", theta: "parameter", h_gradient: "gradient of function, h_gradient(X, theta)", h: "function, h(X, theta)"):
    gradient = (h(X, theta) - y.T) * h_gradient(X, theta) / len(X)
    return gradient


def stochasticGradientDescent(X: "dataset", y: "label", theta0, epsilon, k_max,
                              step, h=computeCostFunc):
    m = len(X)  #size of dataset
    k = 0
    theta = theta0
    f_previous = 0
    f_current = computeCostFunc(X, y, theta, linearTransformation)

    while True:
        if k > k_max:
            break
        for i in range(m):
            theta += -step * computeGradient(X[i], y[i], theta, linearMatrix,
                                               singleLinearTransformation)
        f_previous = f_current
        f_current = computeCostFunc(X, y, theta, linearTransformation)
        if abs(f_current - f_previous) < epsilon:
            break
    return theta, f_current, k
'''

def linearTransformation(theta, X: np.matrix):
    if (X.shape[1] == 1):
        #single vector
        return theta.T * X
    else:
        #vector bundle
        return X * theta

def linearTransformationGradient(theta, X: np.matrix):
    return X


def squareCost(theta, X: np.matrix, y, h):
    return np.sum(np.square(h(theta, X) - y))


def squareCostGradient(theta, X, y, h, h_gradient):
    return 2 * h_gradient(theta, X) * (h(theta, X) - y)


def union_shuffled_copies(arr1, arr2):
    #assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]


def stochasticGradientDescent(X,
                              y,
                              theta0,
                              k_max,
                              step,
                              computeFunc=squareCost,
                              computeGradient=squareCostGradient,
                              h=linearTransformation,
                              h_gradient=linearTransformationGradient,
                              epsilon=0.01):
    '''return optimized point `theta`, current function value and the count of iteration'''
    m = len(X)
    k = 0
    theta = theta0
    f_previous = 0
    f_current = computeFunc(theta, X, y, h) / m
    while (True):
        if k > k_max:
            break
        X, y = union_shuffled_copies(X, y)
        for i in range(m):
            theta -= step * computeGradient(theta, X[i].T, y[i,0], h, h_gradient)
        f_previous = f_current
        f_current = computeFunc(theta, X, y, h)
        if abs(f_current - f_previous) < epsilon:
            break
    return theta, f_current, k
