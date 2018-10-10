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
        if len(parameter) != 2:
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


def getPoolingCoreFunction(poolingType: PoolingType):
    if poolingType == PoolingType.Max:
        return np.max
    elif poolingType == PoolingType.avg:
        return np.average
    #elif poolingType == PoolingType.Lp:
    #return None #TODO
    else:
        #error
        pass


def concatnateInputWithPadding(inputTensor, paddingX: int, paddingY: int):
    lenZ, lenY, lenX = np.shape(inputTensor)
    xPaddingBlock = np.zeros((lenZ, lenY, paddingX))
    result = np.append(xPaddingBlock, inputTensor, axis=2)  # axis=2 is X-axis
    result = np.append(result, xPaddingBlock, axis=2)
    yPaddingBlock = np.zeros((lenZ, paddingY, lenX + 2 * paddingX))
    result = np.append(yPaddingBlock, result, axis=1)
    result = np.append(result, yPaddingBlock, axis=1)
    return result


def pooling(inputTensor,
            kernelSize,
            stride=1,
            padding=0,
            poolingType: PoolingType = PoolingType.Max):
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
        pass

    # concatnate input array with padding
    inputTensor = concatnateInputWithPadding(inputTensor, paddingX, paddingY)
    lenZ, lenY, lenX = np.shape(inputTensor)


    # get pooling function according to poolingType
    poolingCoreFunction = getPoolingCoreFunction(poolingType)

    resultTensor = np.zeros((
        lenZ,
        (lenY - kernelSizeY) // strideY + 1,
        (lenX - kernelSizeX) // strideX + 1,
    ))

    for xResultIndex, xKernelIndex in enumerate(range(0, lenX - kernelSizeX + 1, strideX)):
        for yResultIndex, yKernelIndex in enumerate(range(0, lenY - kernelSizeY + 1, strideY)):
            poolingTensor = inputTensor[:, yKernelIndex:yKernelIndex + kernelSizeY, xKernelIndex:xKernelIndex + kernelSizeX]
            resultTensor[:, yResultIndex, xResultIndex] = poolingCoreFunction(poolingTensor, axis=(1, 2))

    return resultTensor
