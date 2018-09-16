import numpy as np

def gradientDescent(computeGradient, computeFunc, epsilon, step, x0: "initilization of x", k_max: "the max iterations"):
    k = 0  #iteration counter
    f_previous = 0
    f_current = computeFunc(x0)
    x = x0
    while True:
        if k > k_max:
            break
        gradient = computeGradient(x)
        x -= step * gradient
        f_previous = f_current
        f_current = computeFunc(x)
        if abs(f_current - f_previous) < epsilon:
            break
        k += 1
    return x, f_current, k
