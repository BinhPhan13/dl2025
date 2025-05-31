from typing import Literal
import conv
import dense
from arr import Array
from helper import rng
from mat import Matrix
from misc import MCE, Flatten


#%%
def read_data(split: Literal['train', 'test']):
    exts = 'tra' if split == 'train' else 'tes'
    file = 'data/optdigits.' + exts

    shape = 8,8
    max_value = 16

    imgs: list[Array[Matrix]] = []
    labels: list[int] = []
    with open(file) as f:
        for line in f:
            *data, label = line.rsplit(',')
            labels.append(int(label))
            img = Array([Matrix(int(d)/max_value for d in data).on(*shape)])
            imgs.append(img)

    return imgs, labels

#%%
imgs_train, labels_train = read_data('train')
imgs_test, labels_test = read_data('test')

#%%
class Model:
    def __init__(self):
        self.conv_model = conv.Model([
            conv.Layer(1, 2, 3),
            conv.ReLU(),
            conv.Layer(2, 4, 3),
            conv.ReLU(),
            conv.MaxPool(4),
        ])
        self.flat = Flatten()
        self.dense_model = dense.Model([
            dense.Layer(64, 32),
            dense.ReLU(),
            dense.Layer(32, 10),
        ])

    def __call__(self, xs):
        xs = self.conv_model(xs)
        xs = self.flat(xs)
        xs = self.dense_model(xs)
        return xs

    def back(self, grads):
        grads = self.dense_model.back(grads)
        grads = self.flat.back(grads)
        grads = self.conv_model.grad(grads)

    def update(self):
        self.conv_model.update()
        self.dense_model.update()

#%%
rng.seed(7)
model = Model()
loss = MCE()

#%%
def train(lr: float = 0.1, batch: int = 100):
    out = 0.0
    for k, (img, label) in enumerate(zip(imgs_train, labels_train), 1):
        preds = model(img)
        out += loss(preds, label)
        model.back(lr * loss.back())

        if k % batch == 0 or k == len(labels_train):
            print(f"Loss: {out/batch:.5f}")
            model.update()
            out = 0.0

#%%
for i in range(1):
    print("Epoch:", i+1)
    train(0.1, 128)

#%%
acc = 0
for img, label in zip(imgs_test, labels_test):
    _, pred = model(img).max()
    acc += pred == label

print("Accuracy:", acc/len(labels_test))

