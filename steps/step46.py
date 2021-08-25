import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import dezero.functions as F
from dezero import Variable, optimizers
from dezero.models import MLP

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = 0.2
    iters = 10000
    hidden_size = 10

    model = MLP((hidden_size, 1))
    optimizer = optimizers.MomemtumSGD(lr)
    optimizer.setup(model)

    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        if i % 1000 == 0:
            print(loss)
