from math import exp
from arr import Array
from mat import Matrix

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


class Flatten:
    def __call__(self, x: Array[Matrix]) -> Array[float]:
        self.shape = x[0].shape
        out = Array(d for m in x for d in m)
        self.n_out = len(out)
        return out

    def grad(self, grad: Array[float]) -> Array[Matrix]:
        assert len(grad) == self.n_out
        k = self.shape[0] * self.shape[1]
        loop = range(0, self.n_out, k)
        out = Array(
            Matrix(grad[i+j] for j in range(k)).on(*self.shape)
            for i in loop
        )
        return out

