import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from math import *
import time

from dataset import *


def validation(tree, X_val, y_val):
    correct = 0
    y = tree.predict(X_val)
    for i in range(X_val.shape[0]):
        if y[i] == y_val[i]:
            correct += 1
    return correct / X_val.shape[0]


if __name__ == '__main__':
    train_data, train_labels = get_dataset()
    test_data, test_labels = get_dataset(False)

    train_data, test_data = get_grayscale(train_data) / 255, get_grayscale(test_data) / 255

    train_data, test_data = train_data.reshape(-1, 32 * 32), test_data.reshape(-1, 32 * 32)
    max_depth = 5
    max_score = 0
    best_tree = None
    scores = []
    depths = [5, 10, 15, 20, 25, 30]
    ti = time.time()

    for depth in depths:
        print(f"Training depth {depth}")
        tree = DecisionTreeClassifier(criterion='gini', max_depth=depth).fit(train_data, train_labels)
        score = validation(tree, test_data, test_labels)
        if score > max_score:
            max_depth = depth
            max_score = score
            best_tree = tree
        scores.append(score)

    tf = time.time()
    plot_tree(best_tree, max_depth=5)

    plt.plot(scores, depths)
    plt.grid()
    plt.show()
    print(f"Best score: {max_score}")
    print(f"Best depth: {max_depth}")
    print(f"Time: {tf - ti}")

