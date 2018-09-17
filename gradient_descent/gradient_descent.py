import numpy as np

def batchGradientDescent(computeGradient, computeFunc, epsilon, step, x0: "initilization of x", k_max: "the max iterations"):
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


def linearTransformation(X, theta):
    if X.shape[1] == 1 and theta.shape[1] == 1:
        return np.dot(theta.T, X)
    else:
        return np.dot(X, theta)


def linearMatrix(X, theta):
    return theta


def computeCostFunc(X: "dataset", y: "label", theta: "parameter", h: "function, h(X, theta)"):
    return 0.5 * np.sum((h(X, theta) - y.T)** 2)

def computeGradient(X: "dataset", y: "label", theta: "parameter", h_gradient: "gradient of function, h_gradient(X, theta)", h: "function, h(X, theta)"):
    return (h(X, theta) - y.T) * h_gradient(X, theta)


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
            theta.T += -step * computeGradient(X, y, theta, linearMatrix, linearTransformation)
        f_previous = f_current
        f_current = computeCostFunc(X, y, theta, linearMatrix)
        if abs(f_current - f_previous) < epsilon:
            break
    return theta, f_current, k