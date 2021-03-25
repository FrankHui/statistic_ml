# coding: utf-8
"""
重点: 学会手写Baum-Welch的手推
"""
from itertools import product

import numpy as np

from Chapter10.backward import backward
from Chapter10.forward import forward

PENDING = np.array([0])


class HMM:
    def __init__(self, n_state=1, n_output=1, epsilon=1e-3, max_epoch=1000):
        # 状态转移概率矩阵，,shape = j * j， 形如
        # [
        #  [a11, a12, ..., a1j],
        #  [a21, a22, ..., a2j],
        #  ...,
        #  [aj1, a12, ..., ajj]
        # ]
        self.A = np.random.random(size=(n_state, n_state))
        self.A /= self.A.sum(axis=1)[:, None]  # 按行求和，然后按行除以和，确保最终的A每一行的和为1，这样才符合概率和为1

        # 状态->观察的概率矩阵，shape = j * m， 形如
        # [
        #  [b11, b12, ..., b1m],
        #  [b21, b22, ..., b2m],
        #  ...,
        #  [bj1, bj2, ..., bjm],
        # ]
        self.B = np.random.random(size=(n_state, n_output))
        self.B /= self.B.sum(axis=1)[:, None]

        # 初始隐变量的概率分布，shape = (j, ), 形如
        # [p0  p1  p2 ..., pj]
        self.pi = np.ones_like((n_state,)) / n_state

        self.epsilon = epsilon
        self.max_epoch = max_epoch

    def probability(self, O, method='forward'):
        """
        已知λ=(A, B, π)和观测序列O，计算O出现的概率P(O|λ)
        """
        if method == 'forward':
            return forward(self.pi, self.A, self.B, O)
        else:
            return backward(self.pi, self.A, self.B, O)

    def fit(self, O, I):
        """
        正常来说，观测数据是多条O1=(o11, o12, ..., o1s), ..., 按照书上的提示，将这多条拼接成一条大的
        O=(o1, o2, oT)
        """
        O = O.reshape(1, -1)
        I = O.reshape(1, -1)
        if I.size != 0:  # 即有状态序列，使用监督的学习方法
            assert O.shape == I.shape
            # todo: 这里O的shape改了
            # 1. 状态转移概率A的估计，通过频数来估计
            for i in I:
                for i_prev, i_next in zip(i[:-1], i[1:]):
                    self.A[i_prev, i_next] += 1
            self.A /= self.A.sum()
            # 2. 观测概率B的估计
            rows, columns = I.shape
            for row, column in product(range(rows), range(columns)):
                self.B[I[row, column], O[row, column]] += 1
            self.B /= self.B.sum()
            # 3. 估计π
            self.pi = np.unique(I[:, 0], return_counts=True)[1] / I.shape[0]

        else:  # 没有状态序列，则需要用非监督的学习方法——Baum-Welch，背后是EM算法
            for _ in range(self.max_epoch):
                # new_A
                # 1. ξ = (ξ1, ξ2, ..., ξt-1)
                # ξ1形如
                #              下一时刻状态为1    下一时刻状态为2    ...     下一时刻状态为n
                # 此时刻状态为1   p11             p12                     p1n
                # 此时刻状态为2   p21             p22                     p2n
                # ...
                # 此时刻状态为n   pn1             pn2                     pnn
                ksi = []
                gamma = []
                for t in range(len(O[0]) - 1):
                    alpha = forward(self.pi, self.A, self.B, O[0:, t])[0]
                    beta = backward(self.pi, self.A, self.B, O[0:, t])[0]
                    ksi_t = alpha[:, None] * self.A * self.B[:, O[0][t]][None] * beta
                    ksi_t = ksi_t / ksi_t.sum()
                    ksi.append(ksi_t)

                    gamma_t = alpha * beta
                    gamma.append(gamma_t)

                alpha_last = forward(self.pi, self.A, self.B, O[0:, -1])
                beta_last = backward(self.pi, self.A, self.B, O[:, -1])
                gamma_last = alpha_last * beta_last
                gamma.append(gamma_last)

                ksi = np.array(ksi)
                gamma = np.array(gamma)
                new_A = ksi.sum(axis=-1) / gamma.sum(axis=-1)[:, None]

                new_B = 0
                new_pi = 0
                self.A, prev_A = new_A, self.A
                self.B, prev_B = new_B, self.B
                self.pi, prev_pi = new_pi, self.pi
                if np.max(np.abs(self.A - prev_A)) < self.epsilon and np.max(np.abs(self.B - prev_B)) < self.epsilon \
                        and np.max(np.abs(self.pi - prev_pi)) < self.epsilon:
                    break

    def predict(self):
        pass
