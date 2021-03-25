# coding: utf-8
import numpy as np


def backward(M_x, t):
    beta = []
    beta_prev = np.ones(M_x.shape[1])
    beta.append(beta_prev)
    print(f"初始β={beta_prev}")
    for i in range(t - 1, -1, -1):
        beta_next = beta_prev
        M_t = M_x[i + 1]
        beta_prev = np.dot(M_t, beta_next)
        beta.append(beta_prev)
        print(f"β{i}={beta_prev}")
    return beta_prev


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
    beta = backward(M, 3)
    print(beta)


if __name__ == '__main__':
    demo()
