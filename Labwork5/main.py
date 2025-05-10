from nnet import Model, bce_der
from arr import Array

# %%
data = [
    Array([1, 0]),
    Array([1, 1]),
    Array([0, 0]),
    Array([0, 1]),
]
labels = [1, 0, 0, 1]

# %%
model = Model([2, 2, 1])
model.config_wts([
    [-1.5, 1, 1],
    [-0.5, 1, -1],
    [1.5, -1, 1],
])


# %%
def train(lr: float = 0.01):
    for xs, ytrue in zip(data, labels):
        ypred = model(xs)[0]
        grad = lr * bce_der(ytrue, ypred)
        model.grad(Array([grad]))

    model.update()


# %%
for i in range(1000):
    train(1)

# %%
for i in range(len(data)):
    ypred = model(data[i])[0]
    ytrue = labels[i]
    print(ypred, ytrue)

