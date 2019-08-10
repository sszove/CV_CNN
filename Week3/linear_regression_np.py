# -*- coding: utf-8 -*-

"""
Time: 2019.8.4
Author: ssz
Function: Linear Regression
Thought:
Key: batch deal data
Ref:
"""

import numpy as np


def inference(w, b, x_list):

    return np.array(x_list) * w + b


def eval_loss(predict_y_list, y_list):
    p_y_list = np.array(predict_y_list)
    gt_y_list = np.array(y_list)
    # avg cost per example (the 2 in 2*len() doesn't really matter here.
    # But to be consistent with the gradient, I include it)
    return np.dot(np.transpose(p_y_list - gt_y_list), np.array(p_y_list - gt_y_list)) / (2 * len(y_list))


def gradient(predict_y, gt_y, x):
    diff = predict_y - gt_y

    dw = diff * x
    db = diff

    return dw, db


def batch_gradient(predict_y_list, gt_y_list, x_list, w, b, lr):

    batch_size = len(x_list)

    diff = np.array(predict_y_list) - np.array(gt_y_list)
    avg_dw = np.sum(np.multiply(diff, np.array(x_list))) / batch_size
    avg_db = np.sum(diff) / batch_size

    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


def train(x_list, gt_y_list, batch_size, lr, max_iter):

    w = 0
    b = 0

    for i in range(max_iter):
        indexes = np.random.choice(len(x_list), batch_size)

        batch_x_list = [x_list[j] for j in indexes]
        batch_y_list = [gt_y_list[j] for j in indexes]

        batch_predict_y_list = inference(w, b, batch_x_list)

        loss = eval_loss(batch_predict_y_list, batch_y_list)

        w, b = batch_gradient(batch_predict_y_list, batch_y_list, batch_x_list, w, b, lr)

        if i % 100:
            print('w: {}, b: {}'.format(w, b))
            print('loss: {}'.format(loss))

    return w, b


def gen_sample_data():
    w = np.random.randint(0, 10) + np.random.random()		# for noise random.random[0, 1)
    b = np.random.randint(0, 5) + np.random.random()
    num_samples = 10000
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = np.random.randint(0, 100) * np.random.random()
        y = w * x + b + np.random.random() * np.random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w, b


def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 2000, lr, max_iter)

    print('Final w: {}, b: {}'.format(w, b))


if __name__ == '__main__':
    import time
    start = time.time()
    run()
    end = time.time()

    print('Total time used: {}'.format(end - start))