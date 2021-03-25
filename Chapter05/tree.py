# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

INF = np.inf
EPSILON = 1e-2


def _best_split(X, Y):
    """找到最佳的切分特征j和对应的切分点s"""
    rows, cols = X.shape
    if rows <= 1:
        return 0, X[0, 0], 0, 0, 0
    best_j = -1
    best_s = INF
    c1 = INF
    c2 = INF
    best_loss = INF
    for j in range(cols):
        for i in range(rows):
            s = X[i, j]
            R1 = Y[X[:, j] <= s]
            R2 = Y[X[:, j] > s]
            c1_hat = R1.mean()
            c2_hat = R2.mean()
            loss = sum((R1 - c1_hat) ** 2) + sum((R2 - c2_hat) ** 2)
            if loss < best_loss:
                best_j = j
                best_s = s
                c1 = c1_hat
                c2 = c2_hat
                best_loss = loss

    return best_j, best_s, c1, c2, best_loss


class Node:
    def __repr__(self):
        return f"划分特征={self.j} 划分点={self.s} 左标签为{self.c1} 右标签为{self.c2} loss为{self.loss}"

    def __init__(self, j, s, c1, c2, loss, left=None, right=None):
        self.j = j
        self.s = s
        self.c1 = c1
        self.c2 = c2
        self.loss = loss
        self.left = left
        self.right = right
        # self.is_leaf = True


class CartRegressor:
    def __init__(self, max_depth=3):
        self._tree = None
        self.max_depth = max_depth
        self.n_nodes = max_depth * 2 - 1  # Cart是完整二叉树，最大节点数不超过max_depth * 2 - 1

    def fit(self, X, Y, max_depth):
        self.n_nodes = max_depth * 2 - 1
        """递归地对子节点fit"""
        self._tree = Node(*_best_split(X, Y))
        # self._tree = Node(-1, INF, INF, INF)
        n_nodes = 1
        node_list = [(self._tree, X, Y)]  # (节点，节点需要fit的X，Y)
        while node_list:
            node, x, y = node_list.pop(0)
            # print(node)
            # 如果这个节点的loss为0，就不用再细分了
            if node.loss <= EPSILON:
                # node.is_leaf = True
                continue
            part1_index = x[:, node.j] <= node.s
            part2_index = x[:, node.j] > node.s
            x1, y1 = x[part1_index], y[part1_index]
            x2, y2 = x[part2_index], y[part2_index]
            if n_nodes == self.n_nodes:
                continue
            left = Node(*_best_split(x1, y1))
            node_list.append((left, x1, y1))
            node.left = left
            n_nodes += 1
            right = Node(*_best_split(x2, y2))
            node_list.append((right, x2, y2))
            node.right = right
            n_nodes += 1

    def predict_single(self, x):
        node = self._tree
        while node.left or node.right:
            node = node.left if x[node.j] <= node.s else node.right
        return node.c1 if x[node.j] <= node.s else node.c2

    def predict(self, X):
        return np.asarray([self.predict_single(x) for x in X])

    def score(self, X, Y):
        return mean_squared_error(self.predict(X), Y)


def main():
    np.random.seed(0)
    x = np.linspace(-10, 10, 100).reshape((-1, 1))
    y = np.linspace(-20, 20, 100) + np.random.normal(loc=0, scale=3.5, size=(100,))
    # x, y = make_regression(n_samples=500, n_features=2, n_informative=2)
    t = CartRegressor(4)
    df = pd.DataFrame()
    df['x'] = x.reshape((-1,))
    df = df.set_index('x')

    for max_depth in range(2, 8):
        t.fit(x, y, max_depth=max_depth)
        print(f"MAX_DEPTH_{max_depth}: {t.score(x, y)}")
        y_predict = t.predict(x)

        df['MAX_DEPTH_{}'.format(max_depth)] = y_predict

    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, s=10, color='r')

    for max_depth in range(2, 8):
        col_name = 'MAX_DEPTH_{}'.format(max_depth)
        plt.plot(x, df[col_name], label=col_name)
        # plt.show()
    plt.title('Regression Tree')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
