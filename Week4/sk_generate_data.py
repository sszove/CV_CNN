'''
使用sklearn生成四类聚类数据
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmyk'])

datasets = [noisy_circles, noisy_moons, blobs, no_structure]

plt.figure(figsize=(16, 4))
# 画图
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    plt.subplot(1, 4, i_dataset + 1)
    # plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
