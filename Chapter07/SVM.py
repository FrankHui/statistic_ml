# coding: utf-8
"""
SVM
按照解决问题的难度从易到难排序，SVM相关的算法有
线性可分SVM
线性SVM
非线性SVM
由上到下，前序（简单）模型都是后序（复杂）模型的基础、特殊情况，所以只实现非线性，使它兼容前序的模型。

非线性SVM的理解流程：
1. 线性可分SVM的初衷是什么+线性可分SVM参数的计算方法
2. 从硬间隔过渡到软间隔，创造出线性支持SVM，来解决 [近似可分训练集]的分类问题
3. 对于非近似线性可分的训练集，我们的目标是通过映射函数将输入空间映射映射为特征空间，使得训练集在特征空间中近似线性可分，然后应用线性SVM

接下来再针对每个大点做具体理解
1. 线性可分
1.1 SVM的初衷是对空间中正负例画出一个超平面，使得正负例可以被完美隔开，并且不同于感知机，我们还希望无论是每个点都能离这个超平面足够远，
越远我们才会觉得越靠谱。这就有点像及格线60分，成绩越超过60，我们越相信这是个学霸，成绩越低于60，我们越相信这是学渣。
1.2 根据1.1引出函数间隔，进而引出几何间隔
1.3 根据初始最优化的目标的形式，确定最终优化目标是min 1/2 * ||w|| s.t Σ yi(w·xi + b) - 1 >= 0 ，变量是w, b
1.4【重要，不懂拉格朗日就没有必要再看下去了】
    1.4.1 构建拉格朗日函数 L(w,b,α)
    1.4.2 拉格朗日对偶性，确定对偶优化目标是max min L。
        1.4.2.1 min L,变量是w,b。求偏导，得到w关于α的公式，b关于α的公式
        1.4.2.2 代入到L中,得到min L = -1/2 ∑i∑j αi*αj*yi*yj*(xi·xj) + ∑i αi
        1.4.2.3 max [min L]，求得α=(α1, α2, ..., αn) 【这里有个伏笔，当数据量很大的时候，求α其实是非常耗时】
    1.4.3 根据附录定理C.3，我们可以根据对偶问题的最优解α，反过来求得原始问题的最优价w,b

2. 线性不可分，但近似线性可分的情况
2.1 我们对yi(w·xi+b) >= 1的要求放宽一点，允许每个点都能不同程度地达不到这个目标，设置松弛变量ξi，使得 yi(w·xi+b) >= 1 - ξi
2.2 对应优化目标也要对松弛变量加惩罚C，目标变为min 1/2 * ||w|| + Cξ，ξi不为0时，意味着(xi,yi)没有被平面正确分类，否则没有必要松弛。
    所以min 1/2 * ||w|| + Cξ，ξi的后半部分蕴含着少分错点的目标
2.3 同样经过拉格朗日那一套(不过比线性可分的推导过程要复杂),min L = -1/2 ∑i∑j αi*αj*yi*yj*(xi·xj) + ∑i αi s.t. ∑αi yi=0 , 0<=α<=C
2.4 用合页损失来理解线性SVM，这样更容易理解它和感知机的区别。

3. 非线性分类问题(非近似线性可分的训练集)。既然在当前输入空间上，训练集看起来不可分，那能不能通过对空间的映射，使得训练集在映射后
   的空间是线性可分的，或者近似线性可分。
3.1 一个直接的想法，是找到这么一个映射函数φ，但是这个可不好找，怎么就知道φ后的训练集就可分呢？让我们倒过来想，假如我们映射后的训练集可分，那么
    它应该可以用线性SVM搞，那么届时它的目标就是min L = -1/2 ∑i∑j αi*αj*yi*yj*(xi'·xj') +∑i αi，里面的xi',xj'是映射后的，也就是说，
    这里的xi'=φ(xi),xj'=φ(xj)，我们观察到运算单元其实是φ(xi)·φ(xj)，也就是说，我们要是能直接定义出K(xi,xj)=φ(xi)·φ(xj)，也是够用的，
    这个K就是核函数，这种不直接找φ而是找K的方法就是核技巧。
3.2 但是，不能说以xi,xj为变量的二元函数K就是核函数，核函数本意上=两个经过映射后的向量的内积。所以我们需要知道一个K是不是核函数。
    这里有一堆数学知识，按住不表了。
3.3 但即使有3.2，要证明K是核函数还是挺麻烦的，所以一般都是直接应用一些常见的核函数：多项式核函数、高斯核函数核字符串核函数。
3.4 这里我有个问题，好像没有直接证明核函数后的训练集就(近似)线性可分了，大概是拿常用的核函数尝试后，准确率达到一定程度就认为有效吧
3.5 最后我们回到1.4.2.3的伏笔，求α是很麻烦的。好在Platt在1998年提出了SMO(sequential minimal optimization)。实际上，我们手动实现
    SVM，大多数篇幅就是在实现SMO而已。但是不懂前序这些知识，就算是照猫画虎把SMO实现了，笔者认为还不足够
"""

import numpy as np
from functools import partial


def W(K, Y, alpha, i, j):
    """
    i, j分别是第一、第二变量的下标
    """
    _W = .5 * K[i, i] * alpha[i] ** 2 + .5 * K[j, j] * alpha[j] ** 2 + Y[i] * Y[j] * K[i, j] * alpha[i] * alpha[j] - \
         (alpha[i] + alpha[j]) + Y[i] * alpha[i] * ((Y * K[i])[np.r_[:i, i + 1:j:, j + 1:]]).sum() + \
         Y[j] * alpha[j] * (Y * K[j][np.r_[:i, i + 1:j:, j + 1:]]).sum()
    return _W


def SMO(K, Y, alpha, b, epsilon, C):
    """
    SMO要解决如下问题
    min 1/2 ∑i∑j αi*αj*yi*yj*K(xi,xj)-∑i αi
     α
    """
    # 选择变量
    # 先选择第一个变量，选择违反KKT条件最严重的变量作为第一个变量
    pred = np.dot(K, (alpha * Y)) + b  # 书上的g_xi其实就是预测pred
    interval = Y * pred
    error = Y - pred

    # 注意到P129页在“第2个变量的选择”这一节中，最后说明了可能会找不到合适的α2使得目标函数有足够的下降，所以需要遍历寻找直到满足就退出
    # 先在间隔边界上的支持向量点，检验他们是否满足KKT条件（为什么书上说要优先从这里找呢）
    # 记选择的第一个变量是αi,第二个变量αj，即他们的下标分别为i,j
    i_candidate = np.where(
        (0 < alpha < C and interval - 1 > epsilon) or  # todo: 理解这里为什么是 - 1 > epsilon
        (alpha == 0 and interval < 1) or
        (alpha == C and interval > 1)
    )
    for i in i_candidate:
        # 找到第二个变量
        Ei = error[i]
        Ei_minus_Ej = np.abs(error - Ei)
        j_candidate = np.argsort(-Ei_minus_Ej)  # 要对Ei_minus_Ej降序获得下标，np.argsort只支持升序，故排序的时候用相反数
        for j in j_candidate:
            # 更新选定的αi,αj，并计算更新后的αi,αj是否使得子问题W有足够的下降
            # 所以在更新前还得先计算、保存W(αi,αj)
            W_prev = W(K, Y, alpha, i, j)

            # 更新αi,αj
            if Y[i] != Y[j]:
                L = max(0, alpha[j] - alpha[i])
                H = min(C, C + alpha[j] - alpha[i])
            else:
                L = max(0, alpha[j] + alpha[i] - C)
                H = min(C, alpha[j] + alpha[i])

            # 求解未经剪辑的αj_new
            eta = K[i, i] + K[j, j] - 2 * K[i, j]
            Ej = error[j]
            alpha_j_new_unc = alpha[j] + Y[j] * (Ei - Ej) / eta
            # 经剪辑后的αj_new
            if alpha_j_new_unc > H:
                alpha_j_new = H
            elif alpha_j_new_unc >= L:
                alpha_j_new = alpha_j_new_unc
            else:
                alpha_j_new = L
            # 求解αi_new
            alpha_i_new = alpha[i] + Y[i] * Y[j] * (alpha_j_new - alpha[j])
            # 计算是否满足要求
            alpha_new = alpha.copy()
            alpha_new[i] = alpha_i_new
            alpha_new[j] = alpha_j_new
            W_next = W(K, Y, alpha_new, i, j)
            if W_prev - W_next > epsilon:
                b1_new = -Ei - Y[i] *K[i, i] * (alpha_i_new - alpha[i]) - \
                         Y[j] * K[j, i] * (alpha_j_new - alpha[j]) + b
                b2_new = -Ei - Y[i] * K[i, j] * (alpha_i_new - alpha[i]) - \
                         Y[j] * K[j, j] * (alpha_j_new - alpha[j]) + b
                b_new = (b1_new + b2_new) / 2
                return alpha_new, b_new

    return alpha, b


class SVM:
    def __init__(self, C, epsilon, kernel=np.dot):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel  # 默认不经过映射函数，此时核函数就是向量点积而已
        self.w = np.empty((1,))
        self.b = np.random.rand()
        self.alpha = np.empty((1,))

    @staticmethod
    def _data_check(X, Y):
        assert set(Y) == {-1, 1}, "要求训练集中只能用+1,-1的标签"
        assert X.shape[0] == Y.shape[0]

    def fit(self, X, Y):
        self._data_check(X, Y)
        # 根据X来初始化w，α
        self.w = np.empty(X.shape[0])
        self.alpha = np.empty_like(self.w)

        # 先用SMO算法求解α
        K = self.kernel(X, X.T)
        self.alpha, self.b = SMO(K, Y, self.alpha, self.b, self.epsilon, self.C)

        self.w = (self.alpha * Y)[:, None] * X.sum(axis=0)

    def predict(self, X):
        return np.sign(np.dot(X, self.w.T) + self.b)