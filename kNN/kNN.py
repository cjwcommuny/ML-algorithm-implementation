# k-nearest neighbor, kNN

import numpy as np
import cmath
import operator
import heapq


def Lp_distance(x1: np.array, x2: np.array, p=2) -> int:
    if p < 1:
        raise RuntimeError("p is too small")
    elif p == 1:
        return sum(abs(x1 - x2))
    elif p == 2:
        return np.sqrt(sum((x1 - x2)**2))
    elif p == cmath.inf:
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
        self.isVisited = False

    def getValue(self):
        return self.value

    def getLeft(self):
        return self.left

    def setLeft(self, left):
        self.left = left

    def setRight(self, right):
        self.right = right

    def getRight(self):
        return self.right

    def getIsVisited(self):
        return self.isVisited
    
    def setIsVisited(self, flag):
        self.isVisited = flag


def getPartition(X: np.array, i):
    '''return mid value, left array and right array'''
    X = sorted(X, key=lambda x: x[i])
    mid = len(X) // 2
    return X[mid], X[:mid], X[mid+1:]


def constructKdTree(X: np.array):
    return constructKdTreeRecur(X, 0, len(X[0]))


def constructKdTreeRecur(X: np.array, i: int, k: int):
    N = len(X)
    if N == 0:
        return None
    value, left, right = getPartition(X, i)
    node = Node(value, constructKdTreeRecur(left, (i + 1) % k, k), constructKdTreeRecur(right, (i + 1) % k, k))
    return node


def findLeaf(node: Node, x: np.array, i: int, heap: list, k: int, distance):
    if node == None:
        node.setIsVisited(True)
        if len(heap) < k:
            heappush(heap, (distance(x, )))
        return
    dimension = i % len(x)
    if x[dimension] <= node.getValue()[dimension]:
        findLeaf(node.getLeft(), x, i + 1)
    else:
        findLeaf(node.getRight(), x, i + 1)
        

def searchByKdTree(kdTree: Node, x: np.array, k: int):
    kNeighborHeap = []

    pass


def kNN(X: np.array, y: np.array, x: np.array, k: int, distance: "function") -> int:
    pass
