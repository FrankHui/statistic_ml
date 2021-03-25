import numpy as np


def forward(pi, A, B, O):
    """
    pi: 初始状态概率分布(initial state distribution), shape =
    A: 状态转移概率矩阵(state transition matrix)
    B: 状态-->观察概率矩阵(state-->output matrix)
    O: array([观察序列1, 观察序列2, ..., pn])
    任务是计算每条观察序列的概率，形如[p1, p2, ..., pn]
    需要借助α来计算, α形如[α1, α2, ..., αn]
    """
    n_state, _ = A.shape
    assert pi.shape[0] == n_state
    assert B.shape[0] == n_state
    # block: 初始化alpha
    # 1. 每条观察序列的第一个观察值
    o = O[:, 0]
    # 2. 每个初始状态转移到每条观察序列第一个观察值的概率矩阵，形如
    #        第一条序列第一个观测值  第二条序列第一个观测值  ...  第S条序列第一个观测值
    # 状态1   p11                 p12                      p1s
    # 状态2   p21                 p22                      p2s
    # ...
    # 状态n   pn1                 pn2                      pns
    b = B[:, o]
    # 3. 每条观测序列的初始alpha，形如
    #          状态1    状态2   ...  状态n
    # 观测序列1
    # 观测序列2
    # ...
    # 观测序列S
    alpha_next = pi * b.T

    # block: 迭代
    for i in range(1, O.shape[1]):
        alpha_prev = alpha_next
        o = O[:, i]
        b = B[:, o]
        alpha_next = (np.dot(alpha_prev, A)) * b.T

    return alpha_next.sum(axis=1)


def demo():
    pi = np.array([.2, .4, .4])
    A = np.array([
        [.5, .2, .3],
        [.3, .5, .2],
        [.2, .3, .5]
    ])
    B = np.array([
        [.5, .5],
        [.4, .6],
        [.7, .3]
    ])
    O = np.array([
        [0, 1, 0],
        [0, 1, 0],
    ])
    print(f"P(O|λ) = {forward(pi, A, B, O)}")


if __name__ == '__main__':
    demo()