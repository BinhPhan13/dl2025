from math import exp, log
from arr import Array


def sigmoid(x):
    return 1 / (1 + exp(-x))


def loss(ytrue, ws: Array, xs: Array):
    ypred = sum(ws * xs)
    return log(1 + exp(ypred)) - ytrue * ypred


def der(ytrue, ws: Array, xs: Array):
    p = sigmoid(-sum(ws * xs))
    return -xs * ytrue + xs * (1-p)


def gdc(data, lr, tol, max_iter, ws_ini):
    N = len(data)
    L_old = -1

    ws = Array(ws_ini)
    for i in range(max_iter):
        L = sum(loss(y, ws, xs) for xs, y in data)
        if L_old > 0 and abs(L - L_old) < tol: break

        ds = sum(der(y, ws, xs) for xs, y in data) * (1/N)
        ws = ws + -lr * ds

        print(f"{i + 1:<5} {L:<15.5f} {ws}")
        L_old = L

    return ws


def read_data(file: str, sep=","):
    data = []
    with open(file) as f:
        f.readline()
        for line in f:
            *xs, y = line.split(sep)
            xs = Array([1.0, *[float(x) for x in xs]])
            y = float(y)
            data.append([xs, y])
    return data


def acc(data, ws):
    acc = 0
    for xs, y in data:
        ytrue = bool(y)
        ypred = sigmoid(sum(ws*xs)) > 0.5
        acc += ytrue == ypred

    return acc / len(data)

data = read_data("loan2.csv")
ws_ini = [1, 1, 1]
ws = gdc(data, lr=1e-1, tol=1e-3, max_iter=10, ws_ini=ws_ini)


print(acc(data, ws))


