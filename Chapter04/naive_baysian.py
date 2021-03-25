# coding: utf-8
from collections import Counter

"""
朴素贝叶斯
0. 在实现朴素贝叶斯的时候，笔者已经是第N次回顾朴素贝叶斯了，但直到这一次才开始有意识地将它与上一章的感知机做一些对比，
它也给了笔者一些收获。这种与前面的模型/方法做比较的意识，将贯彻整个repository。
1. 朴素贝叶斯的出发点是什么：当已知特征x的条件下，求概率最高的y，所以需要对P(y|x)建模。
而回顾下上一章，感知机的建模是f(x)。
2. 怎么建模: 根据贝叶斯公式:P(y|x)=P(x,y) / P(x)
                              =[P(x|y) * P(y)] / [Σ_{y_i}P(x,y_i)]
                              =[P(x|y) * P(y)] / [Σ_{y_i}P(x|y_i) * P(y_i)]
故需要对P(x|y)和P(y)建模 --> 为什么不能直接对P(y|x)建模，而可以反过来对P(x|y)建模 （其实可以！看看逻辑斯蒂回归)
但这里的任务转化为P(x|y)和P(y)建模后，这个模型必须得具备为P(x|y)和P(y)建模的能力才说得过去！
这就是"朴素贝叶斯法"的贝叶斯。
3. 进一步地，在P(x|y)中，x可能是多维特征，实际上这些特征可能是有关系的。
但朴素贝叶斯做了一个简单的、天真的、朴素的假设：特征之间没有关系。
这就是"朴素贝叶斯"的朴素之处。但是这个朴素的假设有什么用呢 （问题A的答案，下面揭晓）
4. 剩下的问题就是如何为P(x|y)和P(y)建模了
    4.1 使用极大似然估计法估计相应的概率
        4.1.2 P(y)用频数即可
        4.1.3 P(x|y) = P(x1, x2, ..., xn|y)
                     = P(x1|y) * P(x2|y) * ... * P(xn|y) （从上一行到这一行就是基于朴素的"特征之间没有关系"的假设）
                     = [频数(x1, y) / 频数(y)] * [频数(x1, y) / 频数(y)] * ... * [频数(xn, y) / 频数(y)]  
              这里就是朴素假设的用途了，通过这个朴素假设，我们可以通过简单地估计各个P(xi|y)来达到目的
              # todo: P(y|x) = P(y|x1) * P(y|x2) * ... * P(y|xn)???
    4.2 使用贝叶斯估计来避免概率为0的情况
5. 对比下感知机和朴素贝叶斯法。朴素贝叶斯有一步很特别，就是它对P(x,y)建模了，
换句话说，原则上它掌握了(x,y)的生成规律，可以用来生成数据。我们把这类模型叫做生成模型
后续的逻辑斯蒂回归直接对P(y|x)建模，则没有这个生成的过程！
todo: 为什么我们需要对这个特性那么在意？有什么好处吗？
"""


class NaiveBaysian:
    def __init__(self):
        """
        :param features: 特征
        :param labels: label
        """
        self.prior_proba = {}
        self.conditional_proba = []
        self.y_options = {}

    def fit(self, X, Y):
        Y_counts = dict(Counter(Y))
        self.prior_proba = {y: count / len(Y) for y, count in Y_counts.items()}
        self.y_options = set(Y)

        for i in range(len(X[0])):
            X_i = [x[i] for x in X]
            X_i_Y = list(zip(X_i, Y))
            X_i_Y_count = dict(Counter(X_i_Y))
            # P(xi, yi)
            X_i_Y_proba = {x_i_y: count / len(Y) for x_i_y, count in X_i_Y_count.items()}
            # P(xi|yi) = P(xi,yi) / P(yi)
            conditional_proba = {x_i_y: proba / self.prior_proba[x_i_y[1]] for x_i_y, proba in  # x_i_y[1]就是y
                                 X_i_Y_proba.items()}
            self.conditional_proba.append(conditional_proba)
        # 最后self.conditional_proba形如
        # [
        #  第一个特征的条件概率：P(x1|y)={(x1=a, y): p1, (x1=b,y): p2, ..., (x1=z,y): pn},  # 这里的(x1=a,y)代表x1=a|y
        #  第二个特征的条件概率：P(x2|y)={(x1=a, y): p1, (x2=b,y): p2, ..., (x2=z,y): pn},
        #  ...
        #  最后的特征的条件概率：P(xm|y)={(xm=a, y): p1, (xm=b,y): p2, ..., (xm=z,y): pn},
        # ]

    def predict_single(self, x):
        assert len(x) == len(self.conditional_proba)
        y_result = 0
        proba_result = 0
        for y in self.y_options:
            prior_proba = self.prior_proba.get(y, 0)  # 这里要防止训练集中没有出现y
            conditional_proba = 1
            for idx, x_i in enumerate(x):
                conditional_proba *= self.conditional_proba[idx].get((x_i, y), 0)  # 这里要防止训练集中没有出现(x_i, y)
            proba = prior_proba * conditional_proba
            if proba > proba_result:
                proba_result = proba
                y_result = y
        return y_result

    def predict(self, X):
        return [self.predict_single(x) for x in X]


def demo():
    X = [
        [1, 'S'],
        [1, 'M'],
        [1, 'M'],
        [1, 'S'],
        [1, 'S'],
        [2, 'S'],
        [2, 'M'],
        [2, 'M'],
        [2, 'L'],
        [2, 'L'],
        [3, 'L'],
        [3, 'M'],
        [3, 'M'],
        [3, 'L'],
        [3, 'L'],
    ]
    Y = [
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        -1
    ]
    nb = NaiveBaysian()
    nb.fit(X, Y)
    prediction = nb.predict(X)
    print(prediction)
    print(f"正确率为{sum([1 if i == j else 0 for i, j in zip(prediction, Y)]) / len(prediction)}")


if __name__ == '__main__':
    demo()
