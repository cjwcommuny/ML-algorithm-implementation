import numpy as np
from Backpropagation import *
import sys
sys.path.append("../")
import Infrastracture


def test():
    X = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.mat([[0],[1],[1],[0]])
    W = [np.mat([[1,1],[1,1]]), np.mat([[1],[-2]])]
    B = [np.mat([[0], [-1]]), np.mat([0])]
    lam = 0
    ReLU = lambda z: np.multiply(z > 0, z)
    data = Data(X, y)
    param = Parameter(W, B, lam)
    op = Operation(ReLU)
    y_predict = forwardPropagation(data, param, op)
    print(y_predict)
    
test()