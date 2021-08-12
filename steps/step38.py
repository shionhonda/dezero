import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F


if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.reshape((6,))
    y.backward()
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.transpose((1, 0))
    y.backward()
    print(x.grad)
