import numpy as np
import matplotlib.pyplot as plt


def getData():
    pass


def train(X, Y, learningRate=0.000001):
    W = np.array((x.shape[1]), dtype=np.float).reshape(-1, 1)
    B = 0
    numOfIteration = 100
    costs = np.zeros((numOfIteration, 1))
    for i in range(1, numOfIteration):
        Yhat = np.dot(X, W) + B
        r = Yhat - Y
        costs[i] = np.sum(r *r) / 2
        dW = np.dot(X.T, r)
        dB = np.sum(r)
        W -= learningRate * dW
        B -= learningRate * dB
        print("Cost at: ", i, " is ", costs[i - 1])
    return W, B


if __name__ == '__main__':
    numOfPoint = 30
    noise = np.random.normal(0, 1, numOfPoint).reshape(-1, 1)
    x = np.linspace(30, 100, numOfPoint).reshape(-1, 1)
    N = x.shape[0]
    y = 15 * x + 8 + 20 * noise
    plt.scatter(x, y)
    plt.xlabel('mét vuông')
    plt.ylabel('giá')
    # plt.show()
    w, b = train(x, y, learningRate=0.000001)
    print("weight: ", w)
    print("bias: ", b)
    predict = np.dot(x, w) + b
    # plot predict model
    plt.plot((x[0], x[N - 1]), (predict[0], predict[N - 1]), 'r')
    plt.show()
