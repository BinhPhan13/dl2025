from nnet import Model
from arr import Array
from rand import random

def read_cfg(file: str):
    config = []
    with open(file) as f:
        n_layers = int(f.readline())
        for i in range(n_layers+1):
            config.append(int(f.readline()))
    return config


random.seed(42)

config = read_cfg('config.txt')
model = Model(config)
print(model)

x = Array(random.rand() for _ in range(config[0]))
print(x)

print(model(x))

