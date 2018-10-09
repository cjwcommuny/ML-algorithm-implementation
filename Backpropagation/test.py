import sys
import numpy as np
from Backpropagation import Data, Parameter, Operation, forwardPropagation


def test():
    X = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.mat([[0], [1], [1], [0]])
    W = [np.mat([[1, 1], [1, 1]]), np.mat([[1], [-2]])]
    B = [np.mat([[0], [-1]]), np.mat([0])]
    lam = 0
    data = Data(X, y)
    param = Parameter(W, B, lam)
    op = Operation(lambda z: np.multiply(z > 0, z))
    error = forwardPropagation(data, param, op)
    print(error)


test()
