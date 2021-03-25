# coding: utf-8
import numpy as np
from scipy.stats import norm

PENDING = -1


class GMM:
    def __init__(self, k, step=100, epsilon=1e-3):
        self.k = k  # k个高斯分布
        self.alpha = np.ones(k) / k
        # mu形如[mu_1, mu_2, ..., mu_k]
        self.mu = PENDING

        # sigma形如[sigma_1, sigma_2, ..., sigma_k]
        self.sigma = PENDING

        # lambda_matrix形如
        # [
        #  [λ_11, λ_12, ..., λ_1k],
        #  [λ_21, λ_22, ..., λ_2k],
        #  ...,
        #  [λ_n1, λ_n2, ..., λ_nk]
        # ], n是样本的数量，lambda_matrix[j,k]记录的是第k个模型对第j个数据的响应度
        self.lambda_matrix = PENDING

        self.step = step
        self.epsilon = epsilon

    @property
    def k_model(self):
        # P(y|θ) = Σ_{k=1}^{K} alpha_k * norm(
        # 因为norm(loc=self.mu, scale=self.sigma)的shape是(k,)
        # X的shape是(n,)，形如[x1, x2, ..., xn]
        # 而我们希望每个模型都分别n个样本计算概率分布pdf
        # 故需要将X包装成[[x1], [x2], ..., [xn]], 所以用X[:, None]
        return lambda X: self.alpha * norm(loc=self.mu, scale=self.sigma).pdf(X[:, None])

    def fit(self, X):
        """
        GMM学习的是X的分布，是一个无监督学习
        """
        # 根据训练集初始化每个高斯分布的参数μ和σ
        self.mu = np.ones(self.k) * np.mean(X)
        self.sigma = np.ones(self.k) * np.std(X)

        # 开始迭代
        for step in range(self.step):
            # E步：依据当前模型参数，计算分模型k对观测数据y_j的响应度
            self.lambda_matrix = self.k_model(X)
            self.lambda_matrix /= self.lambda_matrix.sum(axis=1)[:, None]

            # M步：计算新一轮的模型参数μ_k, σ_k, α_k
            self.mu = (self.lambda_matrix * X[:, None]).sum(axis=0) / self.lambda_matrix.sum(axis=0)
            self.sigma = (self.lambda_matrix * (X - self.mu) ** 2).sum(axis=0) / self.lambda_matrix.sum(axis=0)
            self.alpha = self.lambda_matrix.sum(axis=0) / X.shape[0]

    def predict(self, X):
        return self.k_model(X).sum()