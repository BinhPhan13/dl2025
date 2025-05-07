class Random:
    def __init__(self):
        self._x = 0

    def seed(self, seed: int):
        self._x = seed

    def rand(self):
        a, c = 13, 17
        m = 37

        self._x = (a * self._x + c)%m
        return self._x / m

random = Random()

