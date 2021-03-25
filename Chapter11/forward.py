# coding: utf-8
"""
为什么CRF算概率时，不像HMM那样，只是计算P(I|O,λ)。而是计算
P(yi|x)和P(yi-1,yi|x)
"""
import numpy as np


def forward(M_x, t):
    """
    根据M矩阵，来计算前向向量αi(yi|x)：即在第i个位置(第i时间步，状态为yi,且
    截止到第i个位置，观察为(x0,x1, ... xi)的概率。
    yi的取值有n个，所以α.shape = (n, )

    注意，书上的M矩阵，指的是从位置i=1,2,..., T+1，各有一个Mi矩阵。
    从而M矩阵由T+1个Mi矩阵组成
    """
    # 书上写的是当y0=start时，才为1，但笔者想不出有什么必要，因为这个start事实上也是虚构头，
    # 这个虚构头的状态为任意一个，我们都无所谓才对，所以这里概率都取为1
    alpha = []
    alpha_next = np.ones(M_x.shape[1])
    alpha.append(alpha_next)
    print(f"初始α={alpha_next}")
    for i in range(t):
        alpha_prev = alpha_next
        M = M_x[i]
        alpha_next = np.dot(alpha_prev, M)
        alpha.append(alpha_next)
        print(f"α{i}={alpha_next}")
    return alpha


def demo():
    M = [
        [
            [.5, .5],
            [.0, .0]
        ],
        [
            [.7, .3],
            [.4, .6]

        ],
        [
            [.2, .8],
            [.5, .5]
        ],
        [
            [.9, .1],
            [.8, .2]
        ]
    ]
    M = np.array(M)
    alpha = forward(M, 3)
    print(alpha)


if __name__ == '__main__':
    demo()
