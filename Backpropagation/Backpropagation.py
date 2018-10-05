import numpy as np
import numpy.matlib

class Node:
    def __init__(self, value: float = 0, parent: Node = None, leftChild: Node = None, sibling: Node = None):
        self.value = value
        self.parent = parent
        self.leftChild = leftChild
        self.sibling = sibling


class Operation:
    def __init__(self, activationFunction, lossFunction, regularFunction):
        self.activationFunction= activationFunction
        self.lossFunction = lossFunction
        self.regularFunction = regularFunction


class Data:
    def __init__(self, X, y):
        self.X: np.array = X
        self.y: np.array = y


class Parameter:
    def __init__(self, W, B, lam):
        self.W: np.array = W
        self.B: np.array = B
        self.lam: float = lam  #factor to adjust regualr term and loss function term 
        

'''
W is 3-d tensor storing weights of the model, W[i] means weight matrix of the ith layer 
B is a matrix storing bias of the model, B[i].T means the bias of ith layer
'''
def forwardPropagation(data: Data, param: Parameter, op: Operation) -> float:
    #extract variable and parameter
    X = data.X
    y = data.y 
    W = param.W #weight
    B = param.B #bias
    lam = param.lam #factor to adjust the loss term and the regular term 
    f = op.activationFunction
    L = op.lossFunction
    omega = op.regularFunction #regular term computation function

    #algorithm
    X_previous = None
    X_current = X
    for k in range(X.shape[0]): #for each layer
        X_previous = X_current
        X_current = f(X_previous @ W[k] + B[k].T) #using broadcast
    y_predict = X_current
    loss = L(y_predict, y) + lam * omega(param)
    return loss
