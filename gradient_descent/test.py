import unittest
import numpy as np
import numpy.matlib
import gradient_descent
import random

def test_squareFunc():
    squareFunc = lambda x: x ** 2
    squareGradient = lambda x: 2 * x
    step = 0.2
    k_max = 200
    x0 = 100
    epsilon = 0.001
    minimalPoint, f_current, k = gradient_descent.batchGradientDescent(squareGradient, squareFunc, epsilon, step, x0, k_max)
    print(minimalPoint, f_current, k)


def test_SGD():
    theta0 = np.mat([1.0, 1.0]).T
    dimension = 2
    m = 10
    X = np.matlib.zeros([m, dimension])
    y = np.matlib.zeros([m, 1])
    theta_original = np.mat([3.0, 5.0]).T
    epsilon = 0.01
    k_max = 100
    step = 0.5
    for i in range(m):
        X[i,0], X[i,1] = random.uniform(-20, 20), random.uniform(-20, 20)
        y[i,0] = gradient_descent.linearTransformation(theta_original, X[i].T)
    theta, f, k = gradient_descent.stochasticGradientDescent(X, y, theta0, k_max, step)
    print(theta, f, k)

#test_squareFunc()
test_SGD()