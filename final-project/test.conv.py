import conv
import dense
from arr import Array
from mat import Matrix
from misc import MCE, Flatten

#%%
conv_model = conv.Model([
    conv.Layer(1, 2, 3),
    conv.ReLU(),
    conv.Layer(2, 4, 3),
    conv.ReLU(),
    conv.MaxPool(4),
])
flat = Flatten()
dense_model = dense.Model([
    dense.Layer(64, 32),
    dense.ReLU(),
    dense.Layer(32, 10),
    dense.ReLU(),
])


#%%
x = Array([Matrix.fill(None, 8, 8)])

x = conv_model(x)
x = flat(x)
x = dense_model(x)

#%%
print(len(x))
print(x)

#%%
loss = MCE()
out = loss(x, 1)
grads = loss.back()

grads = dense_model.back(grads)
grads = flat.back(grads)
grads = conv_model.grad(grads)

#%%
print(len(grads))
print(grads[0].shape)

