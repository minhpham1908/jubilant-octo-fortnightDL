import math
import numpy as np
import matplotlib.pyplot as plt


def getData():
    N = 100  # number of points per class
    d0 = 2  # dimensionality
    C = 3  # number of classes
    X = np.zeros((d0, N * C))  # data matrix (each row = single example)
    y = np.zeros(N * C, dtype='uint8')  # class labels

    for j in range(C):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
        y[ix] = j
    # lets visualize the data:
    # plt.scatter(X[:N, 0], X[:N, 1], c=y[:N], s=40, cmap=plt.cm.Spectral)

    plt.plot(X[0, :N], X[1, :N], 'bs', markersize=5);
    plt.plot(X[0, N:2 * N], X[1, N:2 * N], 'ro', markersize=5);
    plt.plot(X[0, 2 * N:], X[1, 2 * N:], 'g^', markersize=5);
    # plt.axis('off')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    plt.savefig('EX.png', bbox_inches='tight', dpi=600)
    return N, d0, C, X, y


# reLU
def ReLU(z):
    return np.maximum(z, 0)


# ReLU derivative
def ReLU_derivative(z):
    if z <= 0:
        return 0


def softmax(z):
    e_Z = np.exp(z - np.max(z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A


from scipy import sparse


def convert_labels(y, C=3):
    Y = sparse.coo_matrix((np.ones_like(y),
                           (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y


def cost(Y, Yhat):
    return -np.sum(Y * np.log(Yhat)) / Y.shape[1]


class NeuralNetwork():

    def __init__(self, layers, learningRate=0.01) -> None:
        self.layers = layers
        self.learningRate = learningRate
        self.W = []


def plot(W1, b1, W2, b2, X, y):
    # plot the resulting classifier
    h = 0.01
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1.T), W2) + b2.T
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    # plt.scatter(X[:,0], X[:,1], s=40, cmap=plt.cm.Spectral)
    plt.plot(X[0, :N], X[1, :N], 'bs', markersize=5)
    plt.plot(X[0, N:2 * N], X[1, N:2 * N], 'ro', markersize=5)
    plt.plot(X[0, 2 * N:], X[1, 2 * N:], 'g^', markersize=5)

    # plt.axis('off')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    cur_axes = plt.gca()
    # cur_axes.axes.get_xaxis().set_ticks([])
    # cur_axes.axes.get_yaxis().set_ticks([])
    fig.savefig('spiral_net.png', bbox_inches='tight', dpi=600)


def train(X, y):
    d0 = 2
    d1 = h = 100
    d2 = C = 3
    # init Weights and bias
    W1 = 0.01 * np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01 * np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))

    Y = convert_labels(y, C)
    N = X.shape[1]
    learningRate = 1
    count = 0
    tol = 0.001
    loss = []
    while True:
        count += 1
        # feedforward
        Z1 = np.dot(W1.T, X) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2.T, A1) + b2
        Yhat = softmax(Z2)

        if count % 10000 == 0:
            loss.append(cost(Y, Yhat))
            print("iter %d, loss: %f" % (count, loss[-1]))

        # backpropagation
        E2 = (Yhat - Y) / N
        dW2 = np.dot(A1, E2.T)
        db2 = np.sum(E2, axis=1, keepdims=True)
        E1 = np.dot(W2, E2)
        E1[Z1 <= 0] = 0
        dW1 = np.dot(X, E1.T)
        db1 = np.sum(E1, axis=1, keepdims=True)

        # gradient update
        W2 = W2 - learningRate * dW2
        b2 = b2 - learningRate * db2
        W1 = W1 - learningRate * dW1
        b1 = b1 - learningRate * db1
        if count == 10000:
            break

    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    predicted_class = np.argmax(Z2, axis=0)
    print('training accuracy: %.2f %%' % (100 * np.mean(predicted_class == y)))
    plot(W1, b1, W2, b2,X, y)


if __name__ == '__main__':
    N, d0, C, X, y = getData()
    numberOflayer = 2

    train(X, y)
