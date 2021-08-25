import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math

import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
from dezero import DataLoader, optimizers
from dezero.models import MLP

if __name__ == "__main__":
    max_epochs = 5
    batch_size = 100
    hidden_size = 1000

    train_set = dezero.datasets.MNIST(train=True)
    test_set = dezero.datasets.MNIST(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
    optimizer = optimizers.SGD().setup(model)

    if os.path.exists("my_mlp.npz"):
        model.load_weights("my_mlp.npz")

    if dezero.cuda.gpu_enable:
        train_loader.to_gpu()
        model.to_gpu()

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    for epoch in range(max_epochs):
        sum_loss, sum_acc = 0, 0
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        avg_loss = sum_loss / data_size
        avg_acc = sum_acc / data_size
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        print("epoch %d" % (epoch + 1))
        print("train loss %.4f, accuracy %.4f" % (avg_loss, avg_acc))

        with dezero.no_grad():
            sum_loss, sum_acc = 0, 0
            for x, t in train_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

        avg_loss = sum_loss / data_size
        avg_acc = sum_acc / data_size
        test_losses.append(avg_loss)
        test_accs.append(avg_acc)
        print("epoch %d" % (epoch + 1))
        print("test loss %.4f, accuracy %.4f" % (avg_loss, avg_acc))

    model.save_weights("my_mlp.npz")

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(1, max_epochs + 1), train_losses, label="train")
    ax1.plot(range(1, max_epochs + 1), test_losses, label="test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(1, max_epochs + 1), train_accs, label="train")
    ax2.plot(range(1, max_epochs + 1), test_accs, label="test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid()

    fig.tight_layout()
    fig.savefig("./log_52.png")
