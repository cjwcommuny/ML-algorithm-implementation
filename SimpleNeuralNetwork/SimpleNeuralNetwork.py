import numpy as np
import numpy.matlib
import types

class Operation:
    def __init__(self, activationFunction):
        self.activationFunction: types.FunctionType = activationFunction


class Data:
    def __init__(self, X, y):
        self.X: np.mat = X
        self.y: np.mat = y


class Parameter:
    def __init__(self, W, c, w, b):
        self.W: np.mat = W
        self.c: np.mat = c
        self.w: np.mat = w
        self.b: np.mat = b


def ReLU(z: np.mat):
    '''rectified linear unit'''
    return np.multiply(z > 0, z)


def simpleFeedforwardNeuralNetwork(data: Data, op: Operation, param: Parameter):
    '''a feedforward neural network with one hidden layer'''
    X: np.mat = data.X
    #y: np.mat = data.y
    activationFunction: types.FunctionType = op.activationFunction
    W: np.mat = param.W
    c: np.mat = param.c
    w: np.mat = param.w
    b: np.mat = param.b
    return activationFunction(X * W + c.T) * w + b
