"""
输入数据O=(o1, o2, ..., oT)
输出隐变量pi, A, B
todo: new_B公式的含义其实如下
遍历所有时刻t，
1. t时状态为j且t时刻观察为k的概率
2. t是状态为j的概率

"""
import numpy as np

from Chapter10.backward import backward
from Chapter10.forward import forward


def baum_welch(pi, A, B, O, epsilon, max_epoch):
    """
    根据观测数据O来学习、输出隐马尔科夫模型λ=(A, B, π)
    """
    epoch = 0
    T = len(O[0]) - 1
    while epoch < max_epoch:
        print(f"A = \n{A}, \nB = \n{B}, \nπ = \n{pi}")
        epoch += 1
        # 先求ξ_t和γ_t
        # ξ_t形如
        #              下时刻状态为1    下时刻状态为2      ...      下时刻状态为n
        # 此时刻状态为1  p11             p12                      p1n
        # 此时刻状态为2  p21             p22                      p2n
        # ...
        # 此时刻状态为n  pn1             pn2                      pnn

        # γ_t形如
        #  处于状态1   处于状态2  ...   处于状态n
        #  p1         p2             pn

        # 求ξ_t和γ_t需要借助α_t、β_t、β_t+1
        ksi = []
        gamma = []
        # new_B需要知道t时刻状态为j且观察为k的概率，这个量如下计算
        gamma_with_o = []
        for t in range(T):
            alpha_t = forward(pi, A, B, O[:, t])[0]
            beta_t = backward(pi, A, B, O[:, t])[0]
            beta_t_add_1 = backward(pi, A, B, O[:, t + 1])[0]

            ksi_t = alpha_t[:, None] * A * B[:, [t + 1]] * beta_t_add_1[:, None]
            ksi_t = ksi_t / ksi_t.sum()
            ksi.append(ksi_t)

            gamma_t = alpha_t * beta_t
            gamma_t = gamma_t / gamma_t.sum()
            gamma.append(gamma_t)

            # 接下来计算t时刻的gamma_with_o，代表t时刻状态为j且观察为o的概率
            # 形如
            #        观察1  观察2  ... 观察S
            # 状态1
            # 状态2
            # ...
            # 状态n
            output_is_o = np.zeros((B.shape[0],))
            output_is_o[O[:, t][0]] = 1
            gamma_with_o_t = np.dot(gamma_t[:, None], output_is_o[None])
            gamma_with_o.append(gamma_with_o_t)
        ksi = np.array(ksi)
        gamma = np.array(gamma)
        gamma_with_o = np.array(gamma_with_o)

        new_A = ksi.sum(axis=-1) / gamma.sum(axis=-1)[:, None]
        new_B = gamma_with_o.sum(axis=1) / gamma.sum(axis=-1)[:, None]
        new_pi = gamma[0]
        if stop(new_A - A, new_B - B, new_pi - pi, epsilon=epsilon):
            return new_pi, new_A, new_B
        else:
            pi = new_pi
            A = new_A
            B = new_B


def stop(*diffs, epsilon):
    for diff in diffs:
        if abs(diff.max()) < epsilon:
            return True
    return False


def demo():
    pass


if __name__ == '__main__':
    demo()

