from arr import Array
from rand import random
from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))

def bce_der(ytrue: int, ypred: float):
    return (1 - ytrue) / (1 - ypred) - ytrue / ypred


class Node:
    def __init__(self, n_in: int):
        self.ws = Array(random.rand() for _ in range(n_in))
        self.b = 0.0

        self.ws_grads = []
        self.b_grads = []

        self.x = Array([])
        self.a = 0.0

    def config_wts(self, wts: list[float]):
        assert len(wts) == len(self.ws) + 1
        self.b = wts[0]
        self.ws = Array(wts[1:])

    def __call__(self, x: Array):
        assert isinstance(x, Array)
        self.x = x
        self.a = sigmoid(sum(self.ws * x) + self.b)
        return self.a

    def __repr__(self):
        return f"{self.ws}, {self.b}\n"

    def grad(self, grad: float) -> Array:
        sigmoid_grad = self.a * (1 - self.a)
        self.b_grads.append(grad * sigmoid_grad)
        self.ws_grads.append(grad * sigmoid_grad * self.x)
        return grad * sigmoid_grad * self.ws

    def update(self):
        rate = -1/len(self.b_grads)
        self.b = self.b + rate * sum(self.b_grads)
        self.ws = self.ws + rate * sum(self.ws_grads)
        self.b_grads = []
        self.ws_grads = []


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.n_in = n_in
        self.n_out = n_out
        self.nodes = [Node(n_in) for _ in range(n_out)]

    def __call__(self, x: Array):
        return Array(n(x) for n in self.nodes)

    def __repr__(self):
        return f"Layer({self.n_in}, {self.n_out})"

    def grad(self, grads: Array):
        layer_grads = Array(0 for _ in range(self.n_in))
        for n, grad in zip(self.nodes, grads):
            layer_grads = layer_grads + n.grad(grad)
        return layer_grads

    def update(self):
        for n in self.nodes: n.update()


class Model:
    def __init__(self, config: list[int]):
        self.layers: list[Layer] = []
        for i in range(1, len(config)):
            n_in = config[i-1]
            n_out = config[i]
            self.layers.append(Layer(n_in, n_out))

    def config_wts(self, model_wts: list[list[float]]):
        all_nodes: list[Node] = []
        for layer in self.layers: all_nodes.extend(layer.nodes)
        assert len(model_wts) == len(all_nodes)
        for n, wts in zip(all_nodes, model_wts): n.config_wts(wts)

    def __call__(self, x: Array):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return "\n".join(str(n) for n in self.layers)

    def grad(self, grads: Array):
        for layer in reversed(self.layers):
            grads = layer.grad(grads)

    def update(self):
        for layer in self.layers: layer.update()

