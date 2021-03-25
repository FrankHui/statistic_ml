import numpy as np


def backward(pi, A, B, O):
    n_state, _ = A.shape
    assert pi.shape[0] == n_state
    assert B.shape[0] == n_state
    # 初始化β，形如
    #           状态1    状态2    ...    状态n
    # 观测序列1    1       1              1
    # 观测序列2    1       1              1
    # ...
    # 观测序列S    1       1              1
    beta_prev = np.ones((O.shape[0], pi.shape[0]))
    # block: 迭代
    for i in range(O.shape[1] - 1, 0, -1):
        beta_next = beta_prev
        # o形如
        #          观测
        # 观测序列1  o1
        # 观测序列2  o2
        # ...
        # 观测序列S  oS
        o = O[:, i]
        b = B[:, o]
        beta_prev = np.dot(A, (b * beta_next.T)).T

    # 此时得到的beta_prev是指
    o = O[:, 0]
    beta_prev = (pi * B[:, o].T) * beta_prev
    return beta_prev.sum(axis=1)


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
    print(f"P(O|λ) = {backward(pi, A, B, O)}")


if __name__ == '__main__':
    demo()
