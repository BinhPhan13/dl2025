from arr import Array
from rand import random
from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


class Node:
    def __init__(self, n_in: int):
        self.ws = Array(random.rand() for _ in range(n_in))
        self.b = random.rand()

    def __call__(self, x: Array):
        assert isinstance(x, Array)
        return sigmoid(sum(self.ws * x) + self.b)

    def __repr__(self):
        return f"{self.ws}, {self.b}\n"


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.n_in = n_in
        self.n_out = n_out
        self.nodes = [Node(n_in) for _ in range(n_out)]

    def __call__(self, x: Array):
        return Array(n(x) for n in self.nodes)

    def __repr__(self):
        return f"Layer({self.n_in}, {self.n_out})"


class Model:
    def __init__(self, config: list[int]):
        self.layers: list[Layer] = []
        for i in range(1, len(config)):
            n_in = config[i-1]
            n_out = config[i]
            self.layers.append(Layer(n_in, n_out))

    def __call__(self, x: Array):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return "\n".join(str(n) for n in self.layers)

