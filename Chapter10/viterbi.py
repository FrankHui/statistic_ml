# coding: utf-8
"""
维特比算法
任务：
当模型的参数λ=(π,A,B)已知，想要根据观察序列O=(o1, o2, ..., oT),来预测状态序列(i1_, i2_, ..., iT_)【下划线代表它是未知的变量】
每个时刻t都有n种状态的可能，那么状态序列的可能性就有T^n，这个计算量是很大的。
我们需要一种更快捷的办法

假设最优状态序列为I=(i1, i2, ..., iT)，那么它应该具备以下特性：
已知它截止T-1时刻的状态序列为(i1, i2, ..., i_{T-1})。从T-1时刻到T时刻，有状态转移概率iT-1-->iT_的概率，
且iT_还有发射出观察为oT的概率iT_-->oT。iT_有n种可能，其中使得(iT-1-->iT_的概率)*(iT_-->oT的概率)最大的，一定就是iT。
否则，就存在另外一个iT',使得整条序列的概率更大，有矛盾。
这就意味着，算法在求解最后一个时刻T的状态时，答案必须要使得(iT-1-->iT_的概率)*(iT_-->oT的概率)最大。
【关键步骤】现在，让我们把目光往前推一步到T-1，T-1也需要满足这样的条件，T-2也需要，直到t=2时刻(t=1时刻是初始状态)。
因此，我们只需要从t=2开始，每次都求解基于i_{t-1}，分别计算it=1, it=2, ..., it=n的概率最大化的情况
这里举个例子辅助理解：t-1时刻i_{t-1}=1, 2, ..., n, 对应的t时刻it=1概率分别是P11, P12, ..., P1n，如果P1j最大，那么此时应该选择
it=1搭配的i_{t-1}=j，对应最大概率为P1j；同理，计算
it=2搭配的i_{t-1}=k, 对应最大概率为P2k;
...;
it=n搭配的i_{t-1}=m, 对应最大概率为Pnm;

然后递归到下一步，我们可以提炼出一个公式
P_max[t][i] = max(P_max[t-1][1] * a1i * bi(o), P_max[t-1][2] * a2i * bi(o), ..., P_max[t-1][n] * ani * bi(o))
这就是动态规划的公式了。
当然，这个动态规划的任务比一般的动态规划要多一个步骤，因为我们要输出序列，而不是最终最大概率是多少，所以我们还需要记录
第t步it=i时，搭配的i_{t-1}是什么才行。

"""
import numpy as np


def viterbi(pi, A, B, O):
    """
    注意，这里的O是多条观察序列，即O=(O1, O2, ..., Os)，假设每条Oi=(oi1,oi2, ..., oiT)，即每条oi有n_step个时刻
    需要对每条观察序列预测、输出最大概率的状态序列
    """
    A = np.array(A)
    B = np.array(B)
    pi = np.array(pi)
    O = np.array(O)

    # 时刻数（步数（
    _, n_step = O.shape

    # 多条状态序列的shape应该跟O一致
    I = np.empty_like(O)

    # δ代表第t步时，状态分别为1, 2, ..., n的最大概率，形如
    # 第一条观测  状态为1的最大概率 状态为2的最大概率 ... 状态为n的最大概率
    # 第二条观测  状态为1的最大概率 状态为2的最大概率 ... 状态为n的最大概率
    # ...
    # 最后条观测  状态为1的最大概率 状态为2的最大概率 ... 状态为n的最大概率

    # 第0步的delta是根据π和B来初始化的
    delta = pi[None] * B[:, O[:, 0]].T
    psi = np.zeros(shape=(*O.shape, pi.shape[0]))  # psi[k][t][i]代表，第k条观察序列对应的第t步选择状态为i时，搭配t-1的状态

    for t in range(1, n_step):
        psi_t = np.argmax(delta[..., None] * A, axis=1)
        delta = np.max((delta[:, None] * A.T) * B[:, O[:, t]].T[..., None], axis=2)
        psi[:, t] = psi_t

    best_T = np.argmax(delta, axis=1)
    I[:, -1] = best_T
    for t in range(n_step - 2, -1, -1):
        best_t = psi[:, t + 1].take([I[:, t + 1]])
        I[:, t] = best_t
    return I


def demo():
    A = [
        [.5, .2, .3],
        [.3, .5, .2],
        [.2, .3, .5]
    ]

    B = [
        [.5, .5],
        [.4, .6],
        [.7, .3]
    ]

    pi = [.2, .4, .4]

    O = [
        [0, 1, 0],
        [0, 1, 0]
    ]
    print(viterbi(pi, A, B, O))


if __name__ == '__main__':
    demo()
