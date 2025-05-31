from arr import Array
from mat import Matrix


class Module:
    def __call__(self, xs: Array[Matrix]) -> Array[Matrix]: ...
    def back(self, grads: Array[Matrix]) -> Array[Matrix]: ...
    def update(self): return None


class ReLU(Module):
    def __call__(self, xs: Array[Matrix]) -> Array[Matrix]:
        self.out = Array(Matrix(max(0, d) for d in x).on(*x.shape) for x in xs)
        return self.out

    def back(self, grads: Array[Matrix]) -> Array[Matrix]:
        return grads * Array(
            Matrix(float(d > 0) for d in o).on(*o.shape)
            for o in self.out
        )


class _MP:
    def __init__(self, ksize: int = 2):
        self.k = ksize

    def __call__(self, x: Matrix) -> Matrix:
        assert x.nrow % self.k == 0 and x.ncol % self.k == 0
        rs = range(0, x.ncol, self.k)
        cs = range(0, x.nrow, self.k)

        data: list[float] = []
        rcs: list[tuple[int, int]] = []
        for r in rs:
            for c in cs:
                v, (mr, mc) = x[r:r+self.k, c:c+self.k].max()
                data.append(v)
                rcs.append((mr+r, mc+c))

        self.shape = x.shape
        self.rcs = rcs
        self.out = Matrix(data).on(len(rs), len(cs))
        return self.out

    def back(self, grad: Matrix):
        assert grad.shape == self.out.shape
        out = Matrix.fill(0, *self.shape)
        for rc, v in zip(self.rcs, grad): out[rc] = v
        return out


class MaxPool(Module):
    def __init__(self, n_in: int):
        self.nodes = [_MP() for _ in range(n_in)]

    def __call__(self, xs: Array[Matrix]) -> Array[Matrix]:
        return Array(node(x) for node, x in zip(self.nodes, xs))

    def back(self, grads: Array[Matrix]) -> Array[Matrix]:
        return Array(
            node.back(grad) for node, grad in
            zip(self.nodes, grads)
        )


class _Node:
    def __init__(self, n_in: int, ksize: int, pad: int = 1):
        self.ws = Array(Matrix.fill(None, ksize, ksize) for _ in range(n_in))
        self.b = 0.0

        self.ws_grads: list[Array[Matrix]] = []
        self.b_grads : list[float]= []

        self.pad = pad

    def __call__(self, xs: Array[Matrix]) -> Matrix:
        self.xs = Array(m.pad(self.pad, self.pad) for m in xs)
        out = Array(x.conv(w) for x, w in zip(self.xs, self.ws)).sum()
        return out

    def back(self, grad: Matrix) -> Array[Matrix]:
        self.b_grads.append(sum(grad))
        self.ws_grads.append(Array(m.conv(grad) for m in self.xs))

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


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, ksize: int, pad: int = 1):
        self.nodes = [_Node(n_in, ksize, pad) for _ in range(n_out)]

    def __call__(self, xs: Array[Matrix]) -> Array[Matrix]:
        return Array(node(xs) for node in self.nodes)

    def back(self, grads: Array[Matrix]) -> Array[Matrix]:
        return Array(
            node.back(grad)
            for node, grad in zip(self.nodes, grads)
        ).sum()

    def update(self):
        for n in self.nodes: n.update()


class Model:
    def __init__(self, modules: list[Module]):
        self.modules = modules

    def __call__(self, x: Array[Matrix]):
        for module in self.modules:
            x = module(x)
        return x

    def grad(self, grads: Array[Matrix]):
        for module in reversed(self.modules):
            grads = module.back(grads)
        return grads

    def update(self):
        for module in self.modules: module.update()

