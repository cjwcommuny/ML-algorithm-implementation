import numpy as np
import numpy.matlib

class Operation:
    def __init__(
            self,
            activationFunction,
            regularFunction=None,
            lossFunction=None):
        self.activationFunction= activationFunction
        self.lossFunction = lossFunction
        self.regularFunction = regularFunction


class Data:
    def __init__(self, X, y):
        self.X: np.mat = X
        self.y: np.mat = y


class Parameter:
    def __init__(self, W, B, lam):
        self.W: list(np.mat) = W
        self.B: list(np.mat) = B
        self.lam: float = lam  #factor to adjust regualr term and loss function term


'''
W is 3-d tensor storing weights of the model, W[i] means weight matrix of the ith layer 
B is a matrix storing bias of the model, B[i].T means the bias of ith layer
'''
def forwardPropagation(data: Data, param: Parameter, op: Operation):
    #extract variable and parameter
    X = data.X
    y = data.y
    W = param.W #weight
    B = param.B #bias
    lam = param.lam #factor to adjust the loss term and the regular term
    f = op.activationFunction
    L = op.lossFunction
    omega = op.regularFunction #regular term computation function
    depth = len(W)

    #algorithm
    X_previous = None
    X_current = X
    for k in range(depth): #for each layer
        X_previous = X_current
        X_current = f(X_previous * W[k] + B[k].T) #using broadcast
    y_predict = X_current
    #loss = L(y_predict, y) + lam * omega(param)
    #return loss
    return y_predict
