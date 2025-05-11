from typing import Sequence
from arr import Array
from rand import random
from helper import sigmoid

class Act:
    def __call__(self, x: float) -> float: ...
    def grad(self) -> float: ...

class Sigmoid(Act):
    def __call__(self, x: float):
        self.out = sigmoid(x)
        return self.out

    def grad(self):
        return self.out * (1 - self.out)

class ReLU(Act):
    def __call__(self, x: float):
        self.out = max(0, x)
        return self.out

    def grad(self):
        return 1.0 if self.out > 0 else 0.0

ACT_NAME: dict[str, type[Act]] = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
}


_Wts = Sequence[float]
class Node:
    def __init__(self, n_in: int, act: str):
        self.ws = Array(random.rand() for _ in range(n_in))
        self.b = 0.0
        self.x = Array([])

        self.act = ACT_NAME[act]()
        self.ws_grads = []
        self.b_grads = []

    def config_wts(self, wts: _Wts):
        assert len(wts) == len(self.ws) + 1
        self.b = wts[0]
        self.ws = Array(wts[i] for i in range(1, len(wts)))

    def __repr__(self):
        return f"{self.b},\n{self.ws}"

    def __call__(self, x: Array):
        self.x = x
        out = self.act(self.ws@x + self.b)
        return out

    def grad(self, grad: float):
        act_grad = self.act.grad()
        self.b_grads.append(grad * act_grad)
        self.ws_grads.append(grad * act_grad * self.x)
        return grad * act_grad * self.ws

    def update(self):
        rate = -1/len(self.b_grads)
        self.b = self.b + rate * sum(self.b_grads)
        self.ws = self.ws + rate * sum(self.ws_grads)
        self.b_grads = []
        self.ws_grads = []


class Layer:
    def __init__(self, n_in: int, n_out: int, act: str = 'relu'):
        self.n_in = n_in
        self.n_out = n_out
        self.nodes = [Node(n_in, act) for _ in range(n_out)]

    def config_wts(self, layer_wts: Sequence[_Wts]):
        assert len(layer_wts) == len(self.nodes)
        for n, wts in zip(self.nodes, layer_wts):
            n.config_wts(wts)

    def __repr__(self):
        return (
            f"Layer({self.n_in}, {self.n_out})\n" +
            "\n".join(str(n) for n in self.nodes)
        )

    def __call__(self, x: Array):
        return Array(n(x) for n in self.nodes)

    def grad(self, grads: Array):
        assert len(grads) == len(self.nodes)
        out = sum(n.grad(grad) for n, grad in zip(self.nodes, grads))
        assert isinstance(out, Array)
        return out

    def update(self):
        for n in self.nodes: n.update()


class Model:
    @staticmethod
    def config(cfg: list[int], acts: list[str]):
        assert len(cfg) == len(acts) + 1
        return Model([
            Layer(cfg[i - 1], cfg[i], acts[i - 1])
            for i in range(1, len(cfg))
        ])

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def config_wts(self, model_wts: Sequence[_Wts]):
        i = 0
        for layer in self.layers:
            layer_wts = tuple(model_wts[j] for j in range(i, i+layer.n_out))
            layer.config_wts(layer_wts)
            i += layer.n_out

    def __repr__(self):
        return "\n".join(str(n) for n in self.layers)

    def __call__(self, x: Array):
        for layer in self.layers:
            x = layer(x)
        return x

    def grad(self, grads: Array):
        for layer in reversed(self.layers):
            grads = layer.grad(grads)

    def update(self):
        for layer in self.layers: layer.update()

