import numpy as np


def funn(x):
    return x**2

x = np.linspace(0,10,20)

y = np.piecewise(x, [x < 5, x >= 5], [funn, 1])

print(y)