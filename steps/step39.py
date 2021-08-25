import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3, 4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.sum(axis=0, keepdims=True)
    y.backward()
    print(y)
    print(x.grad)
