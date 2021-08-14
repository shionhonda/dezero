import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


if __name__ == "__main__":
    model = MLP((10, 3))
    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2, 0, 1, 0])
    y = model(x)
    loss = F.softmax_cross_entropy(y, t)
    print(loss)
    loss = F.softmax_cross_entropy_simple(y, t)
    print(loss)
