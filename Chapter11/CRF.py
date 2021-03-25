# coding: utf-8
"""
条件随机场(CRF, conditional random field)

1. 初衷：和HMM一样，想要解决有隐变量的序列概率问题，即求解argmax P(I|O,λ=(A,B,π))
                                                   I
2. 区别：
    2.1 HMM
    HMM最重要的预设：预设了每个it只跟i_{t-1}有关，每个ot只跟o_{t-1}有关。第一个假设就是齐次马尔可夫假设。
    换句话说i_{t-1}决定it，o_{t-1}决定ot，前者相当于预设了概率转移矩阵A，后者预设了发射矩阵B。所以HMM需要学习出这两个
    矩阵，再加上初始状态概率分布矩阵π。这三个矩阵学习出来后，意味着HMM已经完全掌握了无论是状态还是观察的生成规律了。所以它就是生成模型。
    这里细细品味一下，正是HMM的这两个假设决定了模型的学习目标，进而决定了模型是生成模型。

    这就是HMM的基本方法论，剩下的难点无非是如何学习出这几个矩阵。
    2.2 CRF(只说最简单的线性CRF)
    CRF则不仅仅是假设每个it只跟i_{t-1}有关，而是假设it跟i_{t-1}和i_{t+1}。也就是说，
    P(it|O,i1,...i_{t-1},i_{t+1},...,iT)=P(it|O,i_{t-1},i_{t+1})
    所以P=(I|O)=（公式11.10)
    # todo: 位置这个概念我，有时候用t，有时候跟书上一致，用的是i，得统一一波
    # todo: 为什么LinearCRF和HMM一样，-在预测都用维特比，在计算概率时都用前向后向
根据例11.1，可以发现，特征函数的定义非常宽泛具体，笔者一开始以为特征函数限定在相对位置，即(前、后位置的标记转移关系)
但后面才发现特征函数可以限定第几个位置，比如t4就限定序列的第二个状态为2和第三个状态为2时才算满足条件
另外，书上的s1,s2,s3,s4都不依赖于具体的
这里用闭包来定义转移特征和状态特征，当然也可以用类定义
统一标识：
X = (X1, X2, ..., Xn)，即n条观察序列。其中Xi = (xi1, xi2, ..., xiT)，即每条观察序列有T个位置。同理：
Y = (Y1, Y2, ..., Yn)，即n条标识序列。其中Yi = (yi1, yi2, ..., yiT)。
每个yit可能的取值有N个
# todo: 一个大问题：看起来,CRF的概率计算、学习、预测都跟x没有任何关系，尤其是根据11.4.1节的对数似然函数，可以发现训练过程中根本用不到x
# todo: 因为fk(yj,xj)的计算过程中，完全用不到xj。(待求证李航老师）但假设现在是给语句分词作标注，我们定义一个状态特征："的"字的标注为O（非实体词），说明
# todo：这种依赖于观察的状态特征是完全合理，笔者擅自按照这种思路来拓宽细化状态特征的定义。
"""
from functools import lru_cache
from functools import reduce
from itertools import product

import numpy as np

from Chapter11.BFGS import BFGS
from Chapter11.backward import backward
from Chapter11.forward import forward

TRANSITION = 'transition'
STATE = 'state'


class FeatureFunc:
    def __init__(self, category, required_y_prev, required_y_next, required_x=None, required_i=None):
        self.category = category
        self.required_y_prev = required_y_prev
        self.required_y_next = required_y_next
        self.required_x = required_x
        self.required_i = required_i

    @lru_cache()
    def cal_single(self, test_y_prev, test_y_next, test_x, test_i):
        """计算给定位置的特征得分"""
        if self.category == TRANSITION:
            if test_y_prev != self.required_y_prev or test_y_next != self.required_y_next:
                return 0
            if self.required_x is not None and test_x != self.required_x:
                return 0
            if self.required_i is not None and test_i != self.required_i:
                return 0
            return 1
        elif self.category == STATE:  # 状态特征只看y_next和位置(如果有要求)
            if test_y_next != self.required_y_prev:
                return 0
            if self.required_i is not None and test_i != self.required_i:
                return 0
            return 1

    @lru_cache()
    def cal_sequence(self, x, y):
        """计算一整个序列的特征得分"""
        score = 0
        start_index = 0 if self.category == STATE else 1
        for test_i in range(start_index, len(x)):
            test_y_prev = y[test_i - 1]
            test_y_next = y[test_i]
            test_x = x[test_i]
            score += self.cal_single(test_y_prev, test_y_next, test_x, test_i)
        return score


class LinearCRF:
    def __init__(self, X, Y, y_option, ff, epsilon):
        """
        :param y_option: 状态的可能值，
        :param X: 观察序列，把多条碾成一条
        """
        # 直接根据这个X,Y来初始化M,α，β，甚至那两个期望值 todo
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        # 计算联合(x,y)的经验概率分布和x的概率分布
        self.x_prob, self.x_y_prob = self._cal_empirical_distribution()
        self.n_sample, self.T = X.shape
        self.y_option = y_option
        self.n = len(self.y_option)  # 状态可能值的数目
        self.ff = ff
        self.w = np.random.dirichlet(size=(len(ff),))
        self.epsilon = epsilon

    def _cal_empirical_distribution(self):
        n_sample = len(self.X)
        x_prob = dict()
        x_y_prob = dict()
        for idx in range(n_sample):
            x, y = tuple(self.X[idx]), tuple(self.Y[idx])
            assert len(x) == len(y), f"第{idx}条样本的状态长度为{len(x)}和输出长度为{len(y)}，不等长"
            x_y = (x, y)
            x_prob[x] = x_prob.get(x, 0) + 1 / n_sample
            x_y_prob[x_y] = x_prob.get(x_y, 0) + 1 / n_sample
        return x_prob, x_y_prob

    def cal_F(self, x, y):
        """
        给定x,y来生成特征矩阵F(y,x)=(f1(y,x),f2(y,x),...,fK(y,x))T
        """
        pass

    def cal_M(self, x):
        """计算给定观察x的前提下的M矩阵"""
        # M是各个时间步上的状态转移矩阵，即M=(M1,M2,...,MT)
        # 形如
        # [
        #   第一个时间步 [                     第一个时间步处于状态1   第一个时间步处于状态2  ...   第一个时间步处于状态n
        #               第零个时间步处于状态1    M11                 M12                      M1n
        #               第零个时间步处于状态2    M21                 M22                      M2n
        #               ...
        #               第零个时间步处于状态n    Mn1                 Mn2                      Mnn
        #              ]
        #   第二个时间步 [                     第二个时间步处于状态1   第二个时间步处于状态2  ...   第二个时间步处于状态n
        #               第一个时间步处于状态1    M11                 M12                      M1n
        #               第一个时间步处于状态2    M21                 M22                      M2n
        #               ...
        #               第一个时间步处于状态n    Mn1                 Mn2                      Mnn
        #              ]
        #  ...
        #   第T+1个时间步 [                     第T+1个时间步处于状态1   第T+1个时间步处于状态2  ...   第T+1个时间步处于状态n
        #                第T个时间步处于状态1    M11                 M12                      M1n
        #                第T个时间步处于状态2    M21                 M22                      M2n
        #                ...
        #                第T个时间步处于状态n    Mn1                 Mn2                      Mnn
        #               ]
        # ]
        # 而Mij=f1(yi,yj,x,1) + f2(yi,yj,x,1) + ...
        # feature_matrix = np.zeros(shape=(self.n, self.n))
        T = len(x)

        M = []
        for test_i in range(T + 1):
            M_t = []
            test_x = x[test_i]
            for test_y_prev, test_y_next in product(range(self.n), range(self.n)):
                score = 0  # 在x下，y_prev, y_next, i在特征函数下的得分
                for w, f in zip(self.w, self.ff):
                    score += w * f(test_y_prev, test_y_next, test_x, test_i)
                M_t.append(score)
            M_t = np.array(M_t).reshape((self.n, self.n))  # 其实到这里，仅仅是书上的W矩阵
            M_t = np.exp(M_t)
            M.append(M_t)
        M = np.array(M)
        return M

    def inference(self, x, y):
        """给定x,y，求Pw(y|x),利用M矩阵"""
        T = len(x)
        M = self.cal_M(x)
        Zw = reduce(np.dot, M)
        numerator = 1
        for i in range(T + 1):
            y_prev = y[i]
            y_next = y[i + 1]
            numerator *= M[i, y_prev, y_next]
        return numerator / Zw

    def fit(self):
        """
        这里用拟牛顿法
        输入：
        1. 原始func: Pw(y|x)，注意这里的func的参数是w，而训练集(X,Y)其实是常数了
        2. func的梯度grad：同样的，grad也是w的梯度
        将1、2传入给BFGS函数，求得最后的w
        :return:
        """

        def loss(w):
            # 先算f(w)的第一项
            term1 = 0
            for x, x_prob in self.x_prob.items():
                exp = 0
                for (x_, y) in self.x_y_prob:  # 在训练集中出现的(x,y)
                    if x_ == x:
                        for w_k, ff_k in zip(w, self.ff):
                            exp += np.exp(ff_k.cal_sequence(x, y))
                term1 += x_prob * np.log(exp)

            term2 = 0
            for (x, y), x_y_prob in self.x_y_prob.items():
                # 计算
                score = 0
                for w_k, ff_k in zip(w, self.ff):
                    score += w * ff_k.cal_sequence(x, y)
                term2 += x_y_prob * score
            cost = term1 - term2
            return cost

        def grad_loss(w):
            self.w = w  # todo
            grad = []
            for w_k, ff_k in zip(self.w, self.ff):
                score = 0
                for (x, y), x_y_prob in self.x_y_prob.items():
                    score += ff_k.cal_sequence(x, y)
                    M = self.cal_M(x)
                    Zm = reduce(np.dot, M)
                    alpha = forward(M, len(x))
                    beta = backward(M, len(x))
            # todo: 还没弄完
            return 0

        self.w, _ = BFGS(loss, grad_loss, self.w, self.epsilon)

    def probability_single(self, x, i):
        """
        :param x: 已知的观察
        :param i: 位置
        :return:
        """
        M = self.cal_M(x)
        # 在当前x下扫描记录α和β
        alpha = []
        alpha_0 = 1  # 书上这里设置的y_0=start时才为1，否则为0，但我没有想出有不为start的必要
        alpha.append(alpha_0)


if __name__ == '__main__':
    pass
