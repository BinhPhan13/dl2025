def f(x):
    return x**2


def f_(x):
    return 2 * x


def main():
    delta = 1e-5
    prev = 1e7
    x = 7
    lr = 0.03

    # for i in range(10):
    while True:
        curr = f(x)
        if abs(curr - prev) < delta:
            break

        x -= lr * f_(x)
        prev = curr
        print(curr)


main()
