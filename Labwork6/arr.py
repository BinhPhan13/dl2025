from typing import Any, Iterable, Sequence, TypeVar, overload
from helper import rng

T = TypeVar('T')
class Array(Sequence[T]):
    def __init__(self, data: Iterable[T]):
        self._data = list(data)

    def __repr__(self):
        return ", ".join(str(x) for x in self)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @overload
    def __getitem__(self, idx: int) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> 'Array[T]': ...
    def __getitem__(self, idx):
        if isinstance(idx, int): return self._data[idx]
        assert isinstance(idx, slice)
        return Array(self._data[i] for i in range(*idx.indices(len(self))))

    @overload
    def __setitem__(self, idx: int, v: Any): ...
    @overload
    def __setitem__(self, idx: slice, v: Iterable[Any]): ...
    def __setitem__(self, idx, v):
        if isinstance(idx, int):
            self._data[idx] = v
            return

        indices = idx.indices(len(self))
        for i, d in zip(indices, v, strict=True):
           self._data[i] = d

    def _scale(self, other):
        return (other for _ in self)

    def __add__(self, other):
        if not isinstance(other, Iterable):
            return Array(a + b for a, b in zip(self, self._scale(other)))
        return Array(a + b for a, b in zip(self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Iterable):
            return Array(a * b for a, b in zip(self, self._scale(other)))
        return Array(a * b for a, b in zip(self, other, strict=True))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if not isinstance(other, Iterable):
            return Array(a - b for a, b in zip(self, self._scale(other)))
        return Array(a - b for a, b in zip(self, other, strict=True))

    def __truediv__(self, other) -> 'Array':
        if not isinstance(other, Iterable):
            return Array(a / b for a, b in zip(self, self._scale(other)))
        return Array(a / b for a, b in zip(self, other, strict=True))

    def __rsub__(self, other):
        if not isinstance(other, Iterable):
            return Array(b - a for a, b in zip(self, self._scale(other)))
        return Array(b - a for a, b in zip(self, other, strict=True))

    def __rtruediv__(self, other) -> 'Array':
        if not isinstance(other, Iterable):
            return Array(b / a for a, b in zip(self, self._scale(other)))
        return Array(b / a for a, b in zip(self, other, strict=True))

    def __neg__(self):
        return 0 - self

    def __matmul__(self, other):
        return self.__mul__(other).sum()

    def sum(self) -> T:
        return sum(x for x in self) # type: ignore

    @staticmethod
    def fill(v: T, n: int) -> 'Array[T]':
        return Array(v for _ in range(n))

    def rand(self, n) -> 'Array[float]':
        return Array(rng.rand() for _ in range(n))

