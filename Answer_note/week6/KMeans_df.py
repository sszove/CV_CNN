import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


def init_centerIDs(df, init_type, k):
    # rows, cols = df.shape
    if init_type == "K-Means":
        # random.choice choices sample
        # DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)[source]
        centerIDs = df.sample(k, replace=False)

    elif init_type == "K-Means++":
        centerIDs = df.sample(1, replace=False)
        for i in range(1, k):
            distance_each_point = np.array(
                [(df['x'] - centerIDs.iloc[j, 0]) ** 2 + (df['y'] - centerIDs.iloc[j, 1]) ** 2
                 for j in range(len(centerIDs))])
            min_distance = np.min(distance_each_point, axis=0)

            sum_distance = min_distance.sum()
            random_sum_distance = random.randint(0, sum_distance)
            current_sum = 0
            for j, x in enumerate(min_distance):
                current_sum += x
                if current_sum > random_sum_distance:
                    centerIDs = centerIDs.append(df.iloc[j+1, :])
                    break
    return centerIDs


def assignment(df, centerIDs):
    distance_each_class = np.array([(df['x']-centerIDs.iloc[i, 0])**2 + (df['y']-centerIDs.iloc[i, 1])**2
                                    for i in range(len(centerIDs))])
    df['class'] = np.argmin(distance_each_class, axis=0)
    return df


def update(df, centerIDs):
    for i in range(len(centerIDs)):
        centerIDs.iloc[i, :] = df[df['class'] == i].mean(0)
    return centerIDs


def main():
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
        })
    k = 3
    max_iterations = 1000
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    centerIDs = init_centerIDs(df, 'K-Means++', k)
    df = assignment(df, centerIDs)
    for i in range(max_iterations):
        centerIDs_pre = centerIDs.copy(deep=True)
        centerIDs = update(df, centerIDs)
        lam = lambda x: x**2
        if (centerIDs-centerIDs_pre).apply(lam).sum().sum() < 1e-1:
            break
        df = assignment(df, centerIDs)

    plt.scatter(centerIDs['x'], centerIDs['y'], color=[colmap[x] for x in range(len(centerIDs))], linewidths=6)
    plt.scatter(df['x'], df['y'], c=[colmap[x] for x in df['class']], alpha=0.5, edgecolors='k')
    plt.show()


if __name__ == "__main__":
    main()
