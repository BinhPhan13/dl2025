from math import exp

def sigmoid(x: float):
    return 1 / (1 + exp(-x))

def bce_der(ytrue: int, ypred: float):
    return (1 - ytrue) / (1 - ypred) - ytrue / ypred

