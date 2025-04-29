def f(w0, w1, xi, yi):
    return 1 / 2 * (w1 * xi + w0 - yi) ** 2


def f_0(w0, w1, xi, yi):
    return w1 * xi + w0 - yi


def f_1(w0, w1, xi, yi):
    return xi * (w1 * xi + w0 - yi)


def gdc(data, lr, tol, max_iter, w0_ini, w1_ini):
    w0 = w0_ini
    w1 = w1_ini
    N = len(data)
    L_old = -1

    for i in range(max_iter):
        L = sum(f(w0, w1, xi, yi) for xi, yi in data) / N
        if L_old > 0 and abs(L - L_old) < tol:
            break

        d0 = sum(f_0(w0, w1, xi, yi) for xi, yi in data) / N
        d1 = sum(f_1(w0, w1, xi, yi) for xi, yi in data) / N

        w0 = w0 - lr * d0
        w1 = w1 - lr * d1

        print(f"{i + 1:<5} {L:<15.5f} {w0=:<10.5f} {w1=:<10.5f}")
        L_old = L

    return w0, w1


def read_data(file: str, sep=","):
    data = []
    with open(file) as f:
        for line in f:
            xi, yi, *_ = line.split(sep)
            data.append([float(xi), float(yi)])
    return data


data = read_data("data.csv")
gdc(data, lr=1e-3, tol=1e-5, max_iter=10, w0_ini=7, w1_ini=10)
