from arr import Array
from helper import rng, sigmoid


class Module:
    def __call__(self, xs: Array[float]) -> Array[float]: ...
    def back(self, grads: Array[float]) -> Array[float]: ...
    def update(self): return None

class Sigmoid(Module):
    def __call__(self, xs: Array[float]) -> Array[float]:
        self.out = Array(sigmoid(x) for x in xs)
        return self.out

    def back(self, grads: Array[float]) -> Array[float]:
        return grads * Array(o*(1-o) for o in self.out)

class ReLU(Module):
    def __call__(self, xs: Array[float]) -> Array[float]:
        self.out = Array(max(0, x) for x in xs)
        return self.out

    def back(self, grads: Array[float]) -> Array[float]:
        return grads * Array(float(o > 0) for o in self.out)


class _Node:
    def __init__(self, n_in: int):
        self.ws = Array(rng.rand() for _ in range(n_in))
        self.b = 0.0

        self.ws_grads: list[Array[float]] = []
        self.b_grads : list[float]= []

    def __call__(self, x: Array[float]) -> float:
        self.x = x
        return self.ws@x + self.b

    def back(self, grad: float) -> Array[float]:
        self.b_grads.append(grad)
        self.ws_grads.append(grad * self.x)
        return grad * self.ws

    def update(self):
        rate = -1/len(self.b_grads)
        self.b = self.b + rate * sum(self.b_grads)
        self.ws: Array[float] = self.ws + rate * sum(self.ws_grads)

        self.b_grads = []
        self.ws_grads = []


class Layer(Module):
    def __init__(self, n_in: int, n_out: int):
        self.nodes = [_Node(n_in) for _ in range(n_out)]

    def __call__(self, x: Array[float]) -> Array[float]:
        return Array(node(x) for node in self.nodes)

    def back(self, grads: Array[float]) -> Array[float]:
        return Array(
            node.back(grad)
            for node, grad in zip(self.nodes, grads)
        ).sum()

    def update(self):
        for n in self.nodes: n.update()


class Model(Module):
    def __init__(self, modules: list[Module]):
        self.modules = modules

    def __call__(self, xs: Array[float]) -> Array[float]:
        for module in self.modules:
            xs = module(xs)
        return xs

    def back(self, grads: Array[float]) -> Array[float]:
        for module in reversed(self.modules):
            grads = module.back(grads)
        return grads

    def update(self):
        for module in self.modules: module.update()

