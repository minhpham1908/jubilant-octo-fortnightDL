import numpy as np
import pandas


def getDataSet():
    x = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                   2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    return x, y


def weightInitialization(n_features):
    w = np.zeros((1, n_features))
    b = 0
    return w, b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def optimize_model(weight, bias, x, y):
    """
    :param weight: weight matrix
    :param bias:
    :param x:
    :param y:
    :return:
    """
    n = x.shape[0]
    activationResult = sigmoid(np.dot(weight, x.T))
    y_t = y.T
    cost = (-1 / n) * (np.sum(y_t * np.log(activationResult) + (1 - y_t) * np.log(1 - activationResult)))

    # gradient descent
    dWeight = (1 / n) * np.dot(x.T, (activationResult - y_t).T)
    dBias = (1 / n) * np.sum(activationResult - y_t)
    grads = {"dWeight": dWeight, "dBias": dBias}
    return grads, cost


def model_predict(weight, bias, x, y, learningRate, noIterations):
    costs = []
    for i in range(noIterations):
        grads, cost = optimize_model(weight, bias, x, y)
        weight = weight - learningRate * grads["dWeight"]
        bias = bias - learningRate * grads["dBias"]
        costs.append(cost)
        if i ==100:
            print("cost at 100th iteration:", end="")
            print(cost)
            break
    coeff = {"weight": weight, "bias": bias}
    return coeff, costs


if __name__ == '__main__':
    x, y = getDataSet();
    x = np.concatenate((np.ones(x.shape[1]), x), axis=0)
    weight, bias = weightInitialization(x.shape[1])
