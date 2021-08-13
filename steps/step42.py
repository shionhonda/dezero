import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        return F.matmul(x, W) + b

    lr = 0.1
    iters = 100
    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)
        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        print(W, b, loss)
