import kNN
import unittest
import numpy as np

class TestKNN(unittest.TestCase):
    def test_distanceFunction(self):
        pass

    def test_constructKdTree(self):
        X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
        kdTree = kNN.constructKdTree(X)
        self.assertEqual(kdTree.value(), np.array([7,2]))
        self.assertEqual(kdTree.getLeft().getValue(), np.array([5,4]))
        #not complete

'''
if __name__ == '__main__':
    unittest.main()
'''

X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
kdTree = kNN.constructKdTree(X)
x = 3