import unittest
import numpy as np
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
    theta0 = np.array([100, 50])
    dimension = 2
    m = 100
    X = np.zeros([m, dimension])
    y = np.zeros([m, 1])
    theta_original = np.array([3, 5])
    epsilon = 0.01
    k_max = 100
    step = 1
    for i in range(m):
        X[i][0], X[i][1] = random.uniform(-100, 100), random.uniform(-100, 100)
        y[i] = gradient_descent.linearTransformation(X[i].T, theta_original)
    theta, f, k = gradient_descent.stochasticGradientDescent(X, y, theta0, epsilon, k_max, step)
    print(theta, f, k)

#test_squareFunc()
test_SGD()