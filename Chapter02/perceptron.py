# coding: utf-8
"""
感知机（以下标号跟书上的章节标号没有关系，后同）
1. 感知机的出发点是什么：找到一个平面尽可能将正实例、负实例分别分到平面两侧，即对y=+1的点，w·x+b>0，反之<0
2. 平面的表示形式：y = w·x + b
3. 1中"尽可能"该如何表达：误分类点的个数越少越好。但这个个数不是w,b的导数，不易优化；改为所有误分类点和平面的总距离尽可能小
4. 误分类点怎么表达：-y(w·x+b)>0
5. 故目标函数：L(w,b)=-Σ_{(x,y)属于误分类点} [y(w·x+b)]
6. 最小化目标函数的方法，求偏导，梯度下降
------到此为止，足以写出代码，但还需要学习以下内容------
7. 算法的收敛性
"""


def sgd_perceptron(w, b, x, y, lr=1):
    """
    根据误分类实例(x,y)更新参数w, b。仅用于感知机
    """
    w = [w_i + lr * x_i * y for w_i, x_i in zip(w, x)]
    b += lr * y
    return w, b


class Perceptron:
    def __init__(self, max_epoch=1000):
        self.w = []
        self.b = 0
        self.max_epoch = max_epoch

    def fit(self, X, Y):
        self.w = [0] * len(X[0])

        epoch = 0
        while True:
            epoch += 1
            all_right = True  # 全都被正确分类
            for x, y in zip(X, Y):
                if sum([w_i * x_i for w_i, x_i in zip(self.w, x)]) * y <= 0:  # 误分类点
                    print(f"误分类点为{(x, y)}")
                    self.w, self.b = sgd_perceptron(self.w, self.b, x, y)
                    all_right = False  # 进入这个if意味着有点没有被正确分类，all_right置为False
                    break
            # 如果经过上述的循环，确实每个点都正确分类，那么可以跳出while循环
            # 或者这个训练集就是无法通过一个超平面分割，那么循环再多次也无法达到all_right，我们设定一个最大循环次数
            if all_right or epoch > self.max_epoch:
                break

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        if sum([w_i * x_i for w_i, x_i in zip(self.w, x)]) + self.b > 0:
            return 1
        else:
            return -1


def demo():
    X = [
        [3, 3],
        [4, 3],
        [1, 1]
    ]
    Y = [
        1,
        1,
        -1
    ]
    clf = Perceptron(max_epoch=20)
    clf.fit(X, Y)
    print(f"w={clf.w}, b={clf.b}")
    print(f"预测结果{clf.predict(X)}")


if __name__ == '__main__':
    demo()
