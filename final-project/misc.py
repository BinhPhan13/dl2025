from math import exp, log
from arr import Array
from mat import Matrix


class BCE:
    def __call__(self, preds: Array[float], label: int) -> float:
        assert label in (0,1)
        self.pred = preds[0]
        self.label = label
        return -(label*log(self.pred) + (1-label)*log(1-self.pred))

    def back(self) -> Array[float]:
        grad = -(self.label / self.pred - (1-self.label)/(1-self.pred))
        return Array([grad])

class MCE:
    def __call__(self, preds: Array[float], label: int) -> float:
        assert label in range(len(preds))
        self.label = label
        preds = Array(exp(p) for p in preds)
        self.preds: Array[float] = preds / preds.sum()
        return -log(self.preds[self.label])

    def back(self) -> Array[float]:
        grad = Array(self.preds)
        grad[self.label] = grad[self.label]-1
        return grad


class Flatten:
    def __call__(self, xs: Array[Matrix]) -> Array[float]:
        self.shape = xs[0].shape
        out = Array(d for x in xs for d in x)
        self.n_out = len(out)
        return out

    def back(self, grads: Array[float]) -> Array[Matrix]:
        assert len(grads) == self.n_out
        k = self.shape[0] * self.shape[1]
        loop = range(0, self.n_out, k)
        out = Array(
            Matrix(grads[i+j] for j in range(k)).on(*self.shape)
            for i in loop
        )
        return out

