import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# def getDataSet():
#     x = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
#                    2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
#     y = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]]).T
#     # x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
#     return x, y


def getDataSet(csvFile="data/iris-data.csv"):
    data = pd.read_csv(csvFile)
    data['class'].replace(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"], inplace=True)
    data = data[data['class']!= "Iris-virginica"]
    data['class'].replace(["Iris-setosa","Iris-versicolor"],[1,0],inplace=True)
    inputData = data.drop(data.columns[[4]], axis=1)
    outputData = data.drop(data.columns[[0,1,2,3]], axis=1)


def weightInitialization(N):
    w = np.zeros((N, 1))
    b = 0
    return w, b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def optimize_model(weight, bias, x, y):
    """
    :param weight: weight matrix N x 1
    :param bias: matrix d x 1
    :param x: matrix d x N
    :param y: matrix d x 1
    :return: grads = {"dWeight": dWeight, "dBias": dBias} and cost
    """
    d = x.shape[0]
    z = np.dot(x, weight) + bias
    activationResult = sigmoid(z)
    # cost matrix 1xd
    cost = (-1 / d) * (np.sum(y * np.log(activationResult) + (1 - y) * np.log(1 - activationResult)))
    # gradient descent
    dWeight = (1 / d) * np.dot(x.T, (activationResult - y))
    dBias = (1 / d) * (np.sum(activationResult - y))
    grads = {"dWeight": dWeight, "dBias": dBias}
    return grads, cost


def model_predict(weight, bias, x, y, learningRate, noIterations, tol=0.0001):
    costs = []
    check_w_after = 20
    for i in range(noIterations):
        grads, cost = optimize_model(weight, bias, x, y)
        weight = weight - learningRate * grads["dWeight"]
        bias = bias - learningRate * grads["dBias"]
        if i % 100 == 0:
            costs.append(cost)
    coeff = {"weight": weight, "bias": bias}
    gradient = grads
    return coeff, gradient, costs


if __name__ == '__main__':
    learningRate = 0.05
    getDataSet()
    # N = x.shape[1]
    # weight, bias = weightInitialization(N)
    # coeff, grads, costs = model_predict(weight, bias, x, y, learningRate, noIterations=60000)
