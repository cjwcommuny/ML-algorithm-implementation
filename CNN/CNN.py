import numpy as np
import scipy
from scipy.signal import convolve
from enum import Enum, unique


@unique
class PoolingType(Enum):
    Max = 0
    Lp = 1
    avg = 2


def getBinaryParameter(parameter):
    '''unpackage the binary parameter'''
    parameter1, parameter2 = 0, 0
    if isinstance(parameter, tuple):
        if len(tuple) != 2:
            pass
            # error
        else:
            parameter1, parameter2 = parameter
    elif isinstance(parameter, int):
        parameter1, parameter2 = parameter, parameter
    else:
        #error
        pass
    return parameter1, parameter2


def maxPooling(input: np.array):
    ''' input is a 3-D tensor'''


def getPoolingCoreFunction(poolingType: PoolingType):
    if poolingType == PoolingType.Max:
        return lambda input: np.max(input)
    elif poolingType == PoolingType.avg:
        return lambda input: np.average(input)
    #elif poolingType == PoolingType.Lp:
        #return None #TODO
    else:
        #error
        pass


def pooling(inputTensor,
            kernelSize,
            stride=1,
            padding=0,
            poolingType: PoolingType = PoolingType.Max):
    paddingValue = 0
    kernelSizeX, kernelSizeY = getBinaryParameter(kernelSize)
    strideX, strideY = getBinaryParameter(stride)
    paddingX, paddingY = getBinaryParameter(padding)

    #convert inputTensor's dim to 3
    inputShapeLen = len(inputTensor.shape)
    if inputShapeLen == 1:
        inputTensor = np.expand_dims(inputTensor, 0)
        inputTensor = np.expand_dims(inputTensor, 0)
        pass
    elif inputShapeLen == 2:
        inputTensor = np.expand_dims(inputTensor, 0)
    elif inputShapeLen == 3:
        pass
    else:
        #error

    #TODO concatnete input array with padding

    poolingCoreFunction = getPoolingCoreFunction(poolingType)  #TODO

    resultTensor = np.array() #TODO
    for xResultIndex, xKernelIndex in range(kernelSizeX + 2 * paddingX - kernelSizeX, step=strideX): #TODO: check boundary and notice that boundary handling
        for yResultIndex, yKernelIndex in range(kernelSizeY + 2 * paddingY - kernelSizeY, step=strideY):
            resultTensor[xResultIndex][yResultIndex] = poolingCoreFunction(inputTensor)

    return resultTensor
