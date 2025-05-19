from math import log
from arr import Array

class BCE:
    def __call__(self, preds: Array[float], label: int) -> float:
        assert label in (0,1)
        self.pred = preds[0]
        self.label = label
        return -(label*log(self.pred) + (1-label)*log(1-self.pred))

    def back(self) -> Array[float]:
        grad = -(self.label / self.pred - (1-self.label)/(1-self.pred))
        return Array([grad])


