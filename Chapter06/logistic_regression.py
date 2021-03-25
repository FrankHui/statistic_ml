# !/Applications/anaconda/envs/4PyCharm/bin/python3.4
# -*- coding: utf-8 -*-
# author: frank
# time  : 2019-06-16 14:40
# file  : logistic_regression.py
import numpy as np
import logging
from collections import Counter
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
"""
逻辑斯蒂回归
1. 以二分类为例子
2. 给定了实例的Feature，判断实例的label是0还是1
3. 两个问题：
3.1 为什么二分类的模型可以是"逻辑斯蒂回归模型"？
    我们预设二分类模型服从伯努利分布，伯努利分布是GLM之一，写出它的GLM形式。
    根据最大熵原则，它是sigmoid形式，或者说是逻辑斯蒂回归模型
    根据最大熵原则，它是sigmoid形式，或者说是逻辑斯蒂回归模型。
3.2 逻辑斯蒂回归的参数估计怎么做？
    极大似然原则，其实跟最大熵原则殊途同归。
换句话说，最大熵原则既决定了模型的"公式"的样子，又决定了参数。

"""


def sigmoid(x):
    activation = 1 / (1 + np.exp(-x))
    return activation


def propagate(features, labels, w, b):
    """
    反向传播梯度下降，此处为了简单起见只做全局梯度下降
    :param features: 特征
    :param labels: 标签
    :param w: 系数
    :param b: 截距
    :return:
    """

    n = features.shape[1]

    # 前向传播
    predictions = sigmoid(np.dot(w.T, features) + b)
    cost = -np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)) / n

    # 反向传播
    d_Z = predictions - labels
    d_w = np.dot(features, d_Z.T) / n
    d_b = np.sum(d_Z) / n

    # w = w - lr * d_w
    # b = b - lr * d_b
    return d_w, d_b, cost


class LogisticRegression:
    """
    初始化
    """
    def __init__(self, lr=0.001, num_epochs=100):
        self.lr = lr
        self.num_epochs = num_epochs

        # 模型的参数
        self.dim = 0
        self.w = np.zeros((0, ))
        self.b = 0

    def fit(self, features, labels):
        """
        拟合、改变参数
        """
        logging.info("开始训练")
        self.dim = features.shape[0]
        self.w = np.ones((self.dim, 1)) * .5

        # 对训练集反向传播
        for epoch in range(self.num_epochs):
            d_w, d_b, cost = propagate(features, labels, self.w, self.b)
            self.w -= d_w * self.lr
            self.b -= d_b * self.lr

            # ==========================================
            # ================  参数衰减  ===============
            # ==========================================
            if epoch == self.num_epochs * .6:
                self.lr *= .5
            if epoch == self.num_epochs * .8:
                self.lr *= .2
            if epoch % 100 == 0:
                logging.info(f"cost = {cost}")
        logging.info(f"===============训练完毕===========")

    def predict(self, instance):
        # p_1 = instance的label是1的概率
        p_1 = sigmoid(np.dot(self.w.T, instance) + self.b)
        return np.where(p_1 > 0.5, 1, 0)


if __name__ == '__main__':

    # 参数设置
    num_cases = 10000
    num_features = 6
    test_lr = 0.1
    test_num_epochs = 5000

    # ==========================================
    # ================  生成数据  ===============
    # ==========================================
    test_features = np.random.rand(num_features, num_cases)
    true_w = (np.arange(1, 7) * np.array([1, -1, 1, -1, 1, -1])).reshape(6, 1)
    true_b = .2
    logging.info(f"true_w=\n{true_w}")
    logging.info(f"true_b={true_b}")

    # w * x + b
    linear_result = np.dot(true_w.T, test_features) + true_b
    # sigmoid(w * x + b)
    test_labels = np.where(sigmoid(linear_result) > 0.5, 1, 0)
    logging.info(f"labels counts are {Counter(test_labels[0])}")

    # 实例化并训练
    LR = LogisticRegression(lr=test_lr, num_epochs=test_num_epochs)
    LR.fit(test_features, test_labels)
    logging.info(f"w=\n{LR.w}")
    logging.info(f"b={LR.b}")

    # accuracy on train data
    train_predictions = LR.predict(test_features)
    result = (train_predictions == test_labels)[0]
    accuracy = Counter(result)[True] / num_cases
    logging.info(f"正确率为{accuracy}")

    # 开始预测
    sample = np.random.rand(num_features, 5)
    true_label = np.where(sigmoid(np.dot(true_w.T, sample) + true_b) > .5, 1, 0)
    logging.info(f"true_label = {true_label}")
    prediction = LR.predict(sample)
    logging.info(f"\nsample=\n{sample}\nprediction={prediction}")
