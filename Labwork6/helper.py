from math import exp
from arr import Array

def sigmoid(x: float):
    return 1 / (1 + exp(-x))

def bce_der(ytrue: int, ypred: float):
    return (1 - ytrue) / (1 - ypred) - ytrue / ypred

def mce_der(ytrue: int, scores: Array[float]):
    # ytrue must in [0, C)
    grad = Array.fill(0, len(scores))
    # softmax
    scores = scores / scores.sum()
    pred = Array(exp(s) for s in scores)[ytrue]
    grad[ytrue] = -1/pred * pred * (1-pred)
    return grad

