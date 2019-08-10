# -*- coding: utf-8 -*-

"""
Time: 2019.8.7
Author: ssz
Function: Logistic Regression
Thought:
Key: batch deal data
Ref:
"""


import numpy as np
from sklearn import preprocessing


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def inference(x_list, w, b):
    # x_list = np.array(x_list)
    # g_z = x_list * w + b
    # h_x = 1 / (1 + np.exp(g_z * -1))
    #
    # return h_x
    predict_y_list = []

    for x in x_list:
        g_z = x * w + b
        h_x = sigmoid(g_z)
        predict_y_list.append(h_x)

    return predict_y_list


def eval_loss(predict_y_list, gt_y_list):

    loss = 0

    batch_size = len(gt_y_list)

    for i in range(batch_size):
        y = gt_y_list[i]
        p_y = predict_y_list[i]
        cost = y * np.log(p_y) + (1 - y) * np.log(1 - p_y)

        loss -= cost

    loss /= batch_size

    return loss


def gradient(predict_y, gt_y, x):
    diff = predict_y - gt_y

    dw = diff * x
    db = diff

    return dw, db


def batch_update_gradient(predict_y_list, gt_y_list, x_list, w, b, lr):

    avg_dw = 0
    avg_db = 0

    batch_size = len(x_list)

    for i in range(batch_size):
        dw, db = gradient(predict_y_list[i], gt_y_list[i], x_list[i])

        avg_dw += dw
        avg_db += db

    avg_dw /= batch_size
    avg_db /= batch_size

    w -= lr * avg_dw
    b -= lr * avg_db

    return w, b


def train(x_list, gt_y_list, lr, batch_size, max_iter):
    w = 0
    b = 0

    normalized_x = preprocessing.normalize([x_list])
    # print(normalized_X.shape)
    x_list = normalized_x[0, :]

    for i in range(max_iter):
        indexes = np.random.choice(len(x_list), batch_size)

        batch_x_list = [x_list[j] for j in indexes]

        batch_y_list = [gt_y_list[j] for j in indexes]

        batch_predict_y_list = inference(batch_x_list, w, b)

        w, b = batch_update_gradient(batch_predict_y_list, batch_y_list, batch_x_list, w, b, lr)

        loss = eval_loss(batch_predict_y_list, batch_y_list)

        if i % 100:
            print('w: {}, b: {}'.format(w, b))
            print('Iter: {}, loss: {}'.format(i, loss))

    return w, b


def gen_sample(w, b, num_samples):

    # num_samples = 10000
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = np.random.randint(-100, 100) * np.random.random()
        z = w * x + b + np.random.random() * np.random.randint(-1, 1)

        if sigmoid(z) >= 0.5:
            y = 1
        else:
            y = 0

        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def run():
    w = np.random.randint(0, 10) + np.random.random()  # for noise random.random[0, 1)
    b = np.random.randint(0, 5) + np.random.random()
    num_samples = 10000
    x_list, gt_y_list = gen_sample(w, b, num_samples)

    lr = 0.001
    batch_size = 500

    max_iter = 10000

    model_w, model_b = train(x_list, gt_y_list, lr, batch_size, max_iter)

    print('Final model: w: {}, b: {}'.format(model_w, model_b))
    print('Ground Truth: w: {}, b: {}'.format(w, b))

    predict_y_list = inference(x_list, model_w, model_b)

    correct = 0
    total = 0

    for i in range(num_samples):
        predict_res = 1 if predict_y_list[i] >= 0.5 else 0
        if predict_res == gt_y_list[i]:
            correct += 1
        total += 1

    print('Training error: {}'.format(1 - correct / total))

    test_x_list, test_y_list = gen_sample(w, b, 1000)

    predict_y_list = inference(test_x_list, model_w, model_b)

    correct = 0
    total = 0

    for i in range(1000):
        predict_res = 1 if predict_y_list[i] >= 0.5 else 0
        if predict_res == test_y_list[i]:
            correct += 1
        total += 1

    print('Testing error: {}'.format(1 - correct / total))


if __name__ == '__main__':
    run()