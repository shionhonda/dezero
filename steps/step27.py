import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np

from dezero import Function, Variable
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return gy * np.cos(x)


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == "__main__":

    x = Variable(np.array(np.pi / 4))
    y = my_sin(x)
    y.backward()

    x.name = "x"
    y.name = "y"
    print(y.data)
    print(x.grad)
    plot_dot_graph(y, verbose=False, to_file="sin.png")
