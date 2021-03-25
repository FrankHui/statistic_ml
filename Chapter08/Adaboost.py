# !/Applications/anaconda/envs/4PyCharm/bin/python3.4
# -*- coding: utf-8 -*-

import numpy as np
from numpy import float

INF = float('inf')


# @lru_cache()
def compute_error(pred, Y, weight):
    return sum(weight * (pred != Y))


class SignClassifier:
    def __repr__(self):
        return ("< " if self.sign == 1 else "> ") + str(self.threshold)

    def __init__(self):
        self.sign = 1
        self.threshold = INF

    def fit(self, X, Y, weight):
        assert len(X) == len(Y) == len(weight)
        X, Y, weight = zip(*sorted(zip(X, Y, weight), key=lambda t: t[0]))
        X, Y, weight = np.array(X), np.array(Y), np.array(weight)
        cost = INF
        for x in np.arange(min(X), max(X), 0.5):
            for sign in [-1, 1]:
                cur_pred = np.array(list(map(lambda t: 1 if t < 0 else -1, X - x))) * sign
                cur_cost = compute_error(cur_pred, Y, weight)
                if cur_cost < cost:
                    cost = cur_cost
                    self.threshold = x
                    self.sign = sign
                if cur_cost == 0:
                    break

    def predict(self, X):
        X = np.array(X)
        return np.array(list(map(lambda t: 1 if t < 0 else -1, X - self.threshold))) * self.sign


class AdaClassifier:
    __slots__ = ['weight', 'n_estimate', 'base_estimate', 'estimate_list', 'am_list']

    def __init__(self, base_estimate, n_estimate):
        self.base_estimate = base_estimate
        self.n_estimate = n_estimate
        # self.weight = 0
        self.estimate_list = []
        self.am_list = []

    def fit(self, X, Y):
        X, Y = np.array(X), np.array(Y)
        weight = np.ones(shape=X.shape) / X.shape[0]  # 初始化权重
        for i in range(self.n_estimate):
            clf = self.base_estimate()
            clf.fit(X, Y, weight)
            self.estimate_list.append(clf)
            # 计算错误率
            em = compute_error(clf.predict(X), Y, weight)
            # 计算指数
            am = .5 * np.log((1 - em) / em)
            self.am_list.append(am)
            # 更新权重
            pred = clf.predict(X)
            exp_list = weight * np.exp(-am * Y * pred)
            Z = sum(exp_list)
            weight = exp_list / Z

    def predict(self, X):
        return np.sign(self.decision_function(X).sum(axis=0))

    def decision_function(self, X):
        return np.array([am * clf.predict(X) for am, clf in zip(self.am_list, self.estimate_list)])

    def score(self, X, Y):
        X, Y = np.array(X), np.array(Y)
        return sum(self.predict(X) == Y) / X.shape[0]


class AdaRegression:
    pass


def main():
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    ada = AdaClassifier(base_estimate=SignClassifier, n_estimate=3)
    ada.fit(X, Y)
    print(ada.decision_function(X))
    print(ada.predict(X))
    print(ada.score(X, Y))


if __name__ == '__main__':
    main()
