from arr import Array
from mat import Matrix

class Act:
    def __call__(self, x: Matrix) -> Matrix: ...
    def grad(self, grad: Matrix) -> Matrix: ...

class ReLU(Act):
    def __call__(self, x: Matrix):
        self.out = Matrix(max(0, d) for d in x).on(*x.shape)
        return self.out

    def grad(self, grad: Matrix):
        return grad * Matrix(d > 0 for d in self.out)

class MaxPool(Act):
    def __init__(self, ksize: int = 2):
        self.ksize = ksize

    def __call__(self, x: Matrix):
        assert x.nrow % self.ksize == 0 and x.ncol % self.ksize == 0
        rs = range(0, x.ncol, self.ksize)
        cs = range(0, x.nrow, self.ksize)

        max_data: list[float] = []
        max_rcs: list[tuple[int, int]] = []
        for r in rs:
            for c in cs:
                v, (mr, mc) = x[r : r + self.ksize, c : c + self.ksize].max()
                max_data.append(v)
                max_rcs.append((mr + r, mc + c))

        self.shape = x.shape
        self.indices = max_rcs
        self.out = Matrix(max_data).on(len(rs), len(cs))
        return self.out

    def grad(self, grad: Matrix):
        assert grad.shape == self.out.shape
        out = Matrix.fill(0, *self.shape)
        for rc, g in zip(self.indices, grad): out[rc] = g
        return out


ACT_NAME: dict[str, type[Act]] = {
    'relu': ReLU,
    'maxpool': MaxPool,
}

class ConvNode:
    def __init__(self, n_in: int, ksize: int, act: str, pad: int = 1):
        self.ws = Array(Matrix.fill(None, ksize, ksize) for _ in range(n_in))
        self.b = 0.0
        self.x = Array([Matrix([])])

        self.act = ACT_NAME[act]()
        self.ws_grads: list[Array[Matrix]] = []
        self.b_grads : list[float]= []

        self.pad = pad

    def __call__(self, x: Array[Matrix]):
        self.x = Array(m.pad(self.pad, self.pad) for m in x)
        out = Array(x.conv(w) for x, w in zip(self.x, self.ws)).sum()
        out = self.act(out + self.b)
        return out

    def grad(self, grad: Matrix) -> Array[Matrix]:
        grad = self.act.grad(grad)
        self.b_grads.append(sum(grad))
        self.ws_grads.append(Array(m.conv(grad) for m in self.x))

        grad = grad.flip()
        return Array(
            w.pad(grad.nrow - 1, grad.ncol - 1).conv(grad)[
                self.pad : -self.pad, self.pad : -self.pad
            ]
            for w in self.ws
        )

    def update(self):
        rate = -1/len(self.b_grads)
        self.b = self.b + rate * sum(self.b_grads)
        self.ws: Array[Matrix] = self.ws + rate * sum(self.ws_grads)
        self.b_grads = []
        self.ws_grads = []


class ConvLayer:
    def __init__(
        self,
        n_in: int,
        n_out: int,
        ksize: int,
        act: str = 'relu',
        pad: int = 1,
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.nodes = [ConvNode(n_in, ksize, act, pad) for _ in range(n_out)]

    def __repr__(self):
        return f"ConvLayer({self.n_in}, {self.n_out})"

    def __call__(self, x: Array[Matrix]):
        return Array(n(x) for n in self.nodes)

    def grad(self, grads: Array[Matrix]):
        assert len(grads) == len(self.nodes)
        return Array(n.grad(g) for n, g in zip(self.nodes, grads)).sum()

    def update(self):
        for n in self.nodes: n.update()


class ConvModel:
    def __init__(self, layers: list[ConvLayer]):
        self.layers = layers

    def __repr__(self):
        return "\n".join(str(n) for n in self.layers)

    def __call__(self, x: Array[Matrix]):
        for layer in self.layers:
            x = layer(x)
        return x

    def grad(self, grads: Array[Matrix]):
        for layer in reversed(self.layers):
            grads = layer.grad(grads)

    def update(self):
        for layer in self.layers: layer.update()

