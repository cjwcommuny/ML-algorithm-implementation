import CNN
import numpy as np

def testConcatnateFunc(inputTensor, padding):
    print("paddingX:", padding[0])
    print("paddingY:", padding[1])
    print("original shape:", np.shape(inputTensor))
    inputTensor = CNN.concatnateInputWithPadding(inputTensor, padding[0], padding[1])
    print("after concatnate:", np.shape(inputTensor))


def testPoolingFunc():
    A = np.random.randint(300, size=(2, 5, 7))
    print("originale shape:", np.shape(A))
    print("first channel:", "\n", A[0])
    print("second channel:", "\n", A[1])

    kernelSize = (3, 2)
    stride = 2
    padding = (2, 1)
    poolingType = CNN.PoolingType.Max
    resultTensor = CNN.pooling(A, kernelSize, stride, padding, poolingType)
    print("new shape:", np.shape(resultTensor))
    print("first channel:", "\n", resultTensor[0])
    print("second channel:", "\n", resultTensor[1])


#a = np.arange(24).reshape((2, 3, 4))
#testConcatnateFunc(a, (2, 1))
testPoolingFunc()