# !/Applications/anaconda/envs/4PyCharm/bin/python3.4
# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from torch.nn import Parameter
from torch.optim import SGD


class LR:
    def __init__(self):
        self.w = torch.tensor(0.)
        self.b = torch.tensor(0.)
        self.step = 100

    def fit(self, X, Y):
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        n_feature = len(X[0])
        n_class = len(Y[0])
        self.w = Parameter(torch.zeros((n_feature, n_class - 1)), requires_grad=True)
        self.b = Parameter(torch.zeros((n_class - 1,)), requires_grad=True)
        optimizer = SGD([self.w, self.b], lr=.1)
        Y = Y.argmax(dim=1)

        for _ in range(self.step):
            optimizer.zero_grad()

            Y_hat_along_label = torch.exp(torch.matmul(X, self.w) + self.b)
            Y_hat_along_label = torch.cat([Y_hat_along_label, torch.ones((len(Y), 1))], 1)
            denominator = Y_hat_along_label.sum(dim=1)
            distribution = Y_hat_along_label / denominator[:, None]
            # loss = torch.nn.CrossEntropyLoss()(Y, distribution)
            loss = torch.nn.NLLLoss()(distribution, Y)
            loss.backward()
            optimizer.step()

    def predict_prob(self, X):
        X = torch.from_numpy(X)
        X = torch.tensor(X, dtype=torch.float32)
        Y_hat_along_label = torch.exp(torch.matmul(X, self.w) + self.b)
        Y_hat_along_label = torch.cat([Y_hat_along_label, torch.ones((len(Y_hat_along_label), 1))], 1)
        denominator = Y_hat_along_label.sum(dim=1)
        distribution = Y_hat_along_label / denominator[:, None]
        return distribution

    def predict_single(self, x):
        x = self.predict_prob(x)
        res = np.zeros_like(x)
        res[x.argmax()] = 1
        return res

    def predict(self, X):
        X = torch.from_numpy(X)
        return np.asarray([self.predict_single(x) for x in X])


def main():
    iris = load_iris()
    X, Y = iris.data, iris.target
    lb = LabelBinarizer()
    Y = lb.fit_transform(Y)
    lr = LR()
    lr.fit(X, Y)
    print(lr.predict_prob(X))
    return lr


if __name__ == '__main__':
    my_lr = main()
