class Array(list):
    def __init__(self, data):
        super().__init__(data)

    def scale(self, v):
        return Array(v for _ in self)

    def __neg__(self):
        return Array(-a for a in self)

    def __add__(self, other):
        if not isinstance(other, Array):
            other = self.scale(other)
        return Array(a + b for a, b in zip(self, other))

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Array):
            other = self.scale(other)
        return Array(a * b for a, b in zip(self, other))

    def __rmul__(self, other):
        return self * other

