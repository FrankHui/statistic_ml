# coding: utf-8
"""
KNN
目的：对于给定目标点A，想要知道A的label
方法：找到训练集中最近的k个点，我们认为这k个点的类别最多的类就是这个点A的类别
也就是说KNN的核心概念只有3个：
1. 距离最近
2. k个
3. 用这k个点的频数最高的label作为目标点A的label预测。

问题：
但是如果遍历搜索所有训练集中的点，来找到最近的距离，这样很耗时。怎么办？
答案：
这就是KDTree的意义，它就是让我们搜索得快一点的办法
所以需要知道，KDTree本质上只是我们为了快速搜索最近k个点的实现手段，它本身不是KNN，只是KDTree这种数据结构具有快速
搜索最近k个点的优点。

"""
from collections import Counter

from Chapter03.kd_tree import KDTree


class KNN:
    """KNN = k nearest neighbour"""

    def __init__(self, k):
        self.k = k
        self.model = None

    def fit(self, X, Y):
        """用KDTree方法来拟合数据，构建模型"""
        self.model = KDTree(X, Y)

    def predict_single(self, x):
        # 找到包含节点的叶节点
        knn_list = self.model.search(x, self.k)
        label_list = [i[1][1] for i in knn_list]
        label_count = Counter(label_list)
        return sorted(label_count.items(), key=lambda t: t[1])[-1][0]

    def predict(self, X):
        return [self.predict_single(x) for x in X]


def demo():
    my_X = [
        [2, 3],
        [5, 4],
        [7, 2],
        [9, 6],
        [8, 1],
        [4, 7]
    ]
    my_Y = [
        0,
        1,
        1,
        0,
        1,
        0
    ]
    knn = KNN(2)
    knn.fit(my_X, my_Y)
    print(knn.model)
    print(knn.predict(my_X))


def demo2():
    my_X = [
        [6.27, 5.5],
        [1.24, -2.86],
        [17.05, -12.79],
        [-6.88, -5.4],
        [-2.96, -0.5],
        [-4.6, -10.55],
        [-4.96, 12.61],
        [1.75, 12.26],
        [7.75, -22.68],
        [10.8, -5.03],
        [15.31, -13.16],
        [7.83, 15.70],
        [14.63, -0.35],
    ]

    my_Y = [
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0
    ]

    knn = KNN(k=1)
    knn.fit(my_X, my_Y)
    print(knn.model)
    print(knn.predict(my_X))


if __name__ == '__main__':
    demo2()
