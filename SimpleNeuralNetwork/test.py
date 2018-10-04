from SimpleNeuralNetwork import *

def test():
    op = Operation(ReLU)
    W = np.mat([[1, 1], [1, 1]])
    c = np.mat([[0], [-1]])
    w = np.mat([[1], [-2]])
    b = 0
    param = Parameter(W, c, w, b)
    X = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.mat([[0],[1],[1],[0]])
    data = Data(X, y)
    y_predict = simpleFeedforwardNeuralNetwork(data, op, param)
    print(y_predict)

test()