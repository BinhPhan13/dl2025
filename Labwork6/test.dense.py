import dense
from arr import Array
from helper import rng
from misc import BCE

# %%
data = [
    Array([1., 0.]),
    Array([1., 1.]),
    Array([0., 0.]),
    Array([0., 1.]),
]
labels = [1, 0, 0, 1]

# %%
rng.seed(7)
model = dense.Model([
    dense.Layer(2, 2),
    dense.Sigmoid(),
    dense.Layer(2, 1),
    dense.Sigmoid(),
])


# %%
loss = BCE()
def train(lr: float = 0.01):
    out = 0.0
    for xs, label in zip(data, labels):
        preds = model(xs)
        out += loss(preds, label)
        model.back(lr * loss.back())

    print(f"Loss: {out/len(labels):.5f}")
    model.update()


# %%
for i in range(1000):
    train(3)

# %%
for i in range(len(data)):
    ypred = model(data[i])[0]
    ytrue = labels[i]
    print(ypred, ytrue)

