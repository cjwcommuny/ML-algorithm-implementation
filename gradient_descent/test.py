import unittest
import numpy as np
import gradient_descent

def test_squareFunc():
    squareFunc = lambda x: x ** 2
    squareGradient = lambda x: 2 * x
    step = 0.2
    k_max = 200
    x0 = 100
    epsilon = 0.001
    minimalPoint, f_current, k = gradient_descent.gradientDescent(squareGradient, squareFunc, epsilon, step, x0, k_max)
    print(minimalPoint, f_current, k)


test_squareFunc()
