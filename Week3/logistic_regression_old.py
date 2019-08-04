# /usr/bin/python
# -*- coding: utf-8 -*-

"""
Time: 2019.08.4
Author: ssz
Function: Logistic Regression
Thought:
Key:
Ref:
"""

import numpy as np
import random


def sigmoid(z):
    result = 1.0 / (1 + np.exp(-z))
    return result


def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (x_len, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (x_len, m),m为样本数
    Y -- 真实标签，shape： (1, m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    # 获取样本数m：
    m = X.shape[1]

    # 前向传播 ：
    A = sigmoid(np.dot(X, w) + b)  # 调用前面写的sigmoid函数
    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    # 反向传播：
    dZ = A - Y
    dw = (np.dot(X, dZ.T)) / m
    db = (np.sum(dZ)) / m

    # 返回梯度：
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    # 定义一个costs数组，存放每若干次迭代后的cost，从而可以画图看看cost的变化趋势：
    costs = []
    # 进行迭代：
    for i in range(num_iterations):
        # 用propagate计算出每次迭代后的cost和梯度：
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        # 用上面得到的梯度来更新参数：
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 每100次迭代，保存一个cost看看：
        if i % 100 == 0:
            costs.append(cost)

        # 这个可以不在意，我们可以每100次把cost打印出来看看，从而随时掌握模型的进展：
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    # 迭代完毕，将最终的各个参数放进字典，并返回：
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


# 经过sigmoid的y值进一步label为二分类0-1
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(X, w) + b)
    for i in range(m):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def logistic_model(X_train, Y_train, W, b, learning_rate=0.1, num_iterations=2000, print_cost=False):
    #梯度下降，迭代求出模型参数：
    params,grads,costs = optimize(W, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    W = params['w']
    b = params['b']

    #用学得的参数进行预测：
    prediction_train = predict(W, b, X_train)

    #计算准确率，在训练集上：
    accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    print("Accuracy on train set:", accuracy_train )

   #为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    d = {"costs": costs,
         "Y_prediction_train" : prediction_train,
         "w": W,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "accuracy_train": accuracy_train
         }
    return d


def gen_sample_data():
    w = random.randint(0, 10) + random.random()  # for noise random.random[0, 1)
    b = random.randint(-5, 5) + random.random()
    Xs = np.random.rand(100) + np.random.randint(-100, 100, size=(1, 100))
    Ys = predict(w, b, Xs)
    return Xs, Ys, w, b


def run():
    Xs, Ys, w, b = gen_sample_data()
    lr = 0.1
    max_iter = 5000
    logistic_model(Xs, Ys, 50, lr, max_iter, True)


if __name__ == '__main__':
    run()

