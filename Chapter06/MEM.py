# coding: utf-8
"""MEM = Maximum entropy model"""
# 最大熵模型原理
# todo: 拉格朗日对偶性，学完这个才知道为什么可以转化为求解max_min_问题
# todo: 理解最大熵模型的最大似然估计等价于最大熵模型的对偶函数
# todo: 牛顿法和拟牛顿法
"""
1. 原则上，MEM需要传入特征函数，如果未传入，则可以简单(朴素)地以每个特征与label的共现作为feature_function
2. 根据特征函数、训练集求得Pw(y|x)，接下来的任务是求得最好的w，当得到了w后，模型就固定了
3. 求得w的方法：IIS
# todo: 这个模型需要遍历、存储x、y的一些性质，这跟生成模型、判别模型有关系吗

feature function定义了哪些数据作为模型的约束，以及数据如何转化为约束
feature function形如
           y1    y2    ...    yt
f1    x1   0/1   0/1          0/1
f2    x2   0/1   0/1          0/1
...
fs    xs   0/1   0/1          0/1
例如
假设在训练集中，x有

"""
import numpy as np
import pandas as pd
from itertools import product


def get_P_XY_and_P_X(X, Y):
    """
    获取联合概率分布和X分布
    联合概率形如
    feature1, feature2, ..., feature, prob_y1, prob_y2, ..., prob_ym
    0       , 0       , ..., 0      , 0.1    , 0.1    , ..., 0
    1       , 0       , ..., 0      , 0.2    , 0      , ..., 0
    ...
    如果总共有10个样本，特征为(1, 0, 0)样本总共有2个，其中有一个y是1，一个y是2，总共可能的y是[1, 2, 3]，那么对应的，它的联合概率如下
    feature1, feature2, feature3, prob_y=1, prob_y=2, prob_y=3
    1       , 0       , 0       , 0.1     , 0.1     , 0
    """
    # 将Y转化成
    XY = np.concatenate(X, Y, axis=1)
    XY_unique, counts = np.unique(XY, axis=1, return_counts=True)
    freq = counts / XY.shape[0]
    df_XY = pd.DataFrame(XY_unique, columns=[f"feature_{i}" for i in range(len(X[0]))] + ['y'])
    df_XY = df_XY.set_index([f"feature_{i}" for i in range(len(X[0]))])['y']
    df_XY = df_XY.unstack().reset_index()

    df_XY.loc[:, 'freq'] = freq
    df_XY = df_XY.groupby([col for col in df_XY.columns if col != 'y']).apply(
        lambda _df: dict(zip(_df['y'], _df['freq']))
    ).reset_index().rename(columns={0: 'distribution'})

    unique_list = [np.unique(X[:, i]) for i in range(len(X[0]))]
    array = np.array(product(*unique_list))
    df = pd.DataFrame(data=array, columns=[f"feature_{i}" for i in range(len(X[0]))])
    zero_distribution = dict(zip(Y.unique(), np.zeros_like(Y.unique())))
    df.loc[: 'distribution_0'] = [zero_distribution for _ in range(len(X[0]))]
    df = pd.merge(df, df_XY, on=df.columns.tolist(), how='left')
    df.loc[: 'distribution'] = np.where()


def get_P_X(X):


class MEM:
    def __init__(self, method='BFGS', epsilon=1e-3):
        """
        """
        self.method = method
        self.epsilon = epsilon
        self.X = np.array([])
        self.Y = np.array([])
        self.p_X = {}
        self.p_XY = {}
        self.n_feature = 1
        self.w = np.random.rand(self.n_feature)
        self.y_options = np.array([])

    def f(self, w):
        pass

    loss_function = f

    def _empirical_joint_distribution(self):
        n_samples = self.X.shape[0]
        X_Y = np.concatenate((self.X, self.Y), axis=1)
        # 以每行作为一个元素计数
        element, freq = np.unique(X_Y, axis=0, return_counts=True)
        element = [tuple(i) for i in element]
        freq /= n_samples
        distribution = dict(zip(element, freq))

        def inner(x, y):
            return distribution[tuple(x) + (y,)]

        return inner

    def get_Pw_y_x(self, w):
        """
        给定参数下的最大熵模型Pw(y|x)
        所谓Pw(y|x)是个概率模型，它可以表示为一个接受x，输出概率分布{y1: p1, y2: p2, ...}的函数（当然也可以有其他表示方法）
        """

        def inner(x):
            numerator_array = np.array([])
            for y in self.y_options:
                numerator = np.exp(w * np.array([f(x, y) for f in self.ffs]))
                numerator_array = np.append(numerator_array, numerator)
            denominator = numerator_array.sum()
            distribution = numerator_array / denominator
            return dict(zip(self.y_options, distribution))

        return inner

    def distribution_matrix(self, X, Y):
        self.

    def fit(self, X, Y):
        X, Y = np.asarray(X), np.asarray(Y)
        # 根据训练数据做一些必要的初始化
        # 1. 获取经验联合分布~P(X,Y)
        empirical_joint_distribution = self._empirical_joint_distribution()
        # 2. 获取给定参数w下的最大熵模型Pw(y|x)

        if self.method == 'IIS':
            pass
        else:
            # 输入特征函数、经验联合分布，目标函数f(w), 梯度函数g(w)
            # 1. 根据特征函数、给定的w，求得最大熵模型Pw_y_x
            # 2. 然后任务是求得最佳的w，将w代进Pw_y_x，就是最终的P_y_x
            # 3. 求解w的方法是
            #       3.1 初始化w、B（正定对称矩阵）
            #       3.2 求梯度g，如果梯度<=epsilon，则停止，否则进入3.3~3.7
            #       3.3 根据B·p = -g,求得p
            #       3.4 一维搜索λ, 使得f(w+pλ)最小
            #       3.5 更新w=w+λp
            #       3.6 更新g,如果g<=epsilon,w_best=w；否则计算新B
            #       3.7 转3.3
            # 备注：另外可以限定循环的次数不超过epochs次

            # 3.1 初始化w, B
            w = np.random.rand(len(self.ffs))
            B = np.eye(len(self.ffs))
            # 3.2 求梯度g
            Pw_y_x = self.get_Pw_y_x(w)

            g = g_w = 0
            for epoch in range(epochs):
                if g <= epsilon:
                    break
