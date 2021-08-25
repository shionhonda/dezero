import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable
from dezero.utils import plot_dot_graph

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = "x"
    y.name = "y"
    y.backward(create_graph=True)

    iters = 5
    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    gx = x.grad
    gx.name = "gx" + str(iters + 1)
    plot_dot_graph(gx, verbose=False, to_file="tanh.png")
