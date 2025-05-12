from rand import random as rng
from typing import Any, Iterable, overload

class Matrix:
    def __init__(self, data: Iterable[Any]):
        self._data = list(data)
        self.nrow = 1
        self.ncol = len(self)

    def on(self, nrow: int, ncol: int):
        assert nrow * ncol == len(self)
        self.nrow = nrow
        self.ncol = ncol
        return self

    def at(self, r: int, c: int):
        return r * self.ncol + c

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @overload
    def __getitem__(self, rc: tuple[int, int]) -> Any: ...
    @overload
    def __getitem__(self, rc: tuple[int | slice, int | slice]) -> 'Matrix': ...
    def __getitem__(self, rc):
        r, c = rc
        if isinstance(r, int) and isinstance(c, int):
            return self._data[self.at(r,c)]

        if isinstance(r, int): r = slice(r-1, r)
        if isinstance(c, int): c = slice(c-1, c)
        rs = range(*r.indices(self.nrow))
        cs = range(*c.indices(self.ncol))

        data = (self._data[self.at(r, c)] for r in rs for c in cs)
        return Matrix(data).on(len(rs), len(cs))

    @overload
    def __setitem__(self, rc: tuple[int, int], v: Any): ...
    @overload
    def __setitem__(self, rc: tuple[int | slice, int | slice], v: Iterable[Any]): ...
    def __setitem__(self, rc, v):
        r, c = rc
        if isinstance(r, int) and isinstance(c, int):
            self._data[self.at(r,c)] = v
            return

        if isinstance(r, int): r = slice(r-1, r)
        if isinstance(c, int): c = slice(c-1, c)
        rs = range(*r.indices(self.nrow))
        cs = range(*c.indices(self.ncol))

        rcs = ((r,c) for r in rs for c in cs)
        for (r,c), d in zip(rcs, v, strict=True):
            self._data[self.at(r,c)] = d


    def __repr__(self):
        return '\n'.join(
            ', '.join(str(self[r, c]) for c in range(self.ncol))
            for r in range(self.nrow)
        )


    def _scale(self, other):
        return (other for _ in self)

    def __add__(self, other):
        if not isinstance(other, Iterable):
            out = Matrix(a + b for a, b in zip(self, self._scale(other)))
        else:
            out = Matrix(a + b for a, b in zip(self, other, strict=True))
        return out.on(*self.shape)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Iterable):
            out = Matrix(a * b for a, b in zip(self, self._scale(other)))
        else:
            out = Matrix(a * b for a, b in zip(self, other, strict=True))
        return out.on(*self.shape)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, Iterable):
            out = Matrix(a - b for a, b in zip(self, self._scale(other)))
        else:
            out = Matrix(a - b for a, b in zip(self, other, strict=True))
        return out.on(*self.shape)

    def __truediv__(self, other):
        if not isinstance(other, Iterable):
            out = Matrix(a / b for a, b in zip(self, self._scale(other)))
        else:
            out = Matrix(a / b for a, b in zip(self, other, strict=True))
        return out.on(*self.shape)

    def __rsub__(self, other):
        if not isinstance(other, Iterable):
            out = Matrix(b - a for a, b in zip(self, self._scale(other)))
        else:
            out = Matrix(b - a for a, b in zip(self, other, strict=True))
        return out.on(*self.shape)

    def __rtruediv__(self, other):
        if not isinstance(other, Iterable):
            out = Matrix(b / a for a, b in zip(self, self._scale(other)))
        else:
            out = Matrix(b / a for a, b in zip(self, other, strict=True))
        return out.on(*self.shape)

    def __neg__(self):
        return 0 - self

    def conv(self, other: "Matrix"):
        assert other.nrow <= self.nrow and other.ncol <= self.ncol
        rs = range(self.nrow - other.nrow + 1)
        cs = range(self.ncol - other.ncol + 1)

        return Matrix(
            sum(self[r : r + other.nrow, c : c + other.ncol] * other)
            for r in rs for c in cs
        ).on(len(rs), len(cs))

    def pad(self, rpad: int = 1, cpad: int = 1, v: Any = 0):
        nrow = self.nrow + rpad*2
        ncol = self.ncol + cpad*2

        out = Matrix.fill(v, nrow, ncol)
        out[rpad: rpad + self.nrow, cpad : cpad + self.ncol] = self
        return out

    def flip(self):
        return Matrix(reversed(self._data)).on(*self.shape)


    @property
    def shape(self):
        return self.nrow, self.ncol

    @staticmethod
    def fill(v: Any, nr: int, nc: int):
        loop = (v for r in range(nr) for c in range(nc))
        if v is not None:
            return Matrix(loop).on(nr, nc)
        return Matrix(rng.rand() for _ in loop).on(nr,nc)

