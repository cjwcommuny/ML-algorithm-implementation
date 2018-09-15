# k-nearest neighbor, kNN

import numpy as np
from cmath import *
import operator


def Lp_distance(x1: np.array, x2: np.array, p=2) -> int:
    if p < 1:
        raise RuntimeError("p is too small")
    elif p == 1:
        return sum(abs(x1 - x2))
    elif p == 2:
        return np.sqrt(sum((x1 - x2)**2))
    elif p == inf:
        return max(x1 - x2)
    else:
        return (sum((x1 - x2)**p))**(1/p)


def getKNN(X: np.array, y: np.array, x: np.array, k: int,
           distance: "function") -> np.array:
    pass


def makeDecision(neighbors):
    '''Given the k-nearest neighbours, determine the label of given vector'''
    classDict = {}
    for neighbor in neighbors:
        if not neighbor in classDict.keys():
            classDict[neighbor] = 1
        else:
            classDict[neighbor] += 1
    return max(classDict.items(), key=operator.itemgetter(1))[0] #get the key value of the max value


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def getValue(self):
        return self.value

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right


def getPartition(X: np.array, i):
    '''return mid value, left array and right array'''
    sorted(X, key=lambda x: x[i])
    mid = X.len() / 2
    return X[mid], X[:mid], X[mid+1:]


def constructKdTree(X: np.array):
    N = len(X)
    k = len(X[0])
    for i in range(N):
        l = i % k #the dimension of partition 
        value, left, right = getPartition(X, l)
        node = 
    pass


def kNN(X: np.array, y: np.array, x: np.array, k: int, distance: "function") -> int:
    pass
