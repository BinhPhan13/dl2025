from math import exp

def sigmoid(x: float):
    return 1 / (1 + exp(-x))

class Random:
    def __init__(self):
        self._x = 0
        # ANSI C
        self.a = 1103515245
        self.c = 12345
        self.m = 2**31

    def seed(self, seed: int):
        self._x = seed % self.m

    def _update(self):
        self._x = (self.a * self._x + self.c) % self.m

    def rand(self):
        self._update()
        out = self._x / self.m
        return -1 + out * 2

rng = Random()

