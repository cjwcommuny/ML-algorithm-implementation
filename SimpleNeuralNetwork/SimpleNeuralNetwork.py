import numpy as np
import numpy.matlib
import types

class Operation:
    activationFunction: types.FunctionType = None

    def __init__(self, activationFunction):
        self.activationFunction = activationFunction

class Data:
    X: np.mat = None
    y: np.mat = None
    def __init__(self, X, y):
        self.X = X
        self.y = y

class Parameter:
    W: np.mat = None #weight
    c: np.mat = None #bias
    w: np.mat = None  #weight
    b: np.mat = None  #bias

    def __init__(self, W, c, w, b):
        self.W = W
        self.c = c
        self.w = w
        self.b = b


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
