# !/Applications/anaconda/envs/4PyCharm/bin/python3.4
# -*- coding: utf-8 -*-
import math
from collections import Counter, deque
from functools import reduce


def calculate_entropy(labels):
    """
    计算label集的熵
    :param labels: list
    :return: 熵: float
    """
    total = len(labels)

    # 每个类的数量，计算熵的时候，类本身并不重要，重要的是每个类各种的数量/比例
    counter_of_every_class = Counter(labels).values()
    # 每个类的比例
    scale_of_every_class = map(lambda x: x / total, counter_of_every_class)
    res = sum(map(lambda i: -i * math.log(i), scale_of_every_class))
    # my_print(res)
    return res


class _Node:
    """
    树的节点，每个节点用来fit一个特征
    """

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.label = None
        self.idx_feature = None  # idx_feature用来记载这个节点选择了哪个特征分量来拆分树
        self.child_dict = {}  # 选择了特征分量，按照这个特征分量的n个取值划分出若干子集合，这个节点的子节点分别一个子集合

    def fit(self, features, labels):
        """
        :param features: X = 样本 * [特征0, 特征1, ……]
        :param labels: Y = 样本 * label
        :return:
        """

        assert len(features) == len(labels), "X和Y的个数不一致"

        # 当labels都为一样的，这个节点就有自己的label了，没有子节点
        if len(set(labels)) == 1:
            self.label = labels[0]
            return

        # 如果已经没有特征的话，跟上面一样
        num_features = len(features[0])  # 特征的个数
        if not num_features:
            self.label = Counter(labels).most_common(1)[0][0]  # 计数，然后选最多的那个
            return

        """
        计算每个特征列的信息熵
        """
        cols = [[sample[idx] for sample in features] for idx in range(num_features)]
        entropy_list = []
        for col in cols:  # 对于每个特征列
            set_of_types_in_col = set(col)
            total_entropy = 0
            for s in set_of_types_in_col:  # 对于这个特征列的每个取值
                subset = [label for c, label in zip(col, labels) if c == s]
                total_entropy += calculate_entropy(subset) * (len(subset) / len(labels))
            entropy_list.append(total_entropy)

        # 挑选出【使得分割后集合的信息熵最少】的特征
        min_idx, min_entropy = reduce(lambda x, y: x if x[1] < y[1] else y, enumerate(entropy_list))

        """
        这个特征会使得互信息最大（信息不确定性的减少最多）
        如果连这个互信息都达不到epsilon，我们认为每个特征都提供不了多少信息，那再继续分支也没有什么价值
        所以直接取占比最高的类作为这个节点的label
        """
        if calculate_entropy(labels) - min_entropy < self.epsilon:
            self.label = Counter(labels).most_common(1)[0][0]
            return

        # 否则就挑选这个特征
        self.idx_feature = min_idx

        # 挑选之后，按照这个特征的n个取值，它会产生n个子节点
        # 同时我们需要划分集合
        # 每个子节点(child)对应处理一个子集(sub_feature和sub_labels)
        set_n_value = set([sample[min_idx] for sample in features])  # n个取值的集合，形如{0, 1}、{1, 2, 3}这样
        for value in set_n_value:
            sub_features = []  # 子特征集
            sub_labels = []  # 子label集
            for sample, label in zip(features, labels):
                if sample[min_idx] == value:
                    sub_features.append(sample[:min_idx] + sample[min_idx + 1:])
                    sub_labels.append(label)
            child = _Node(epsilon=self.epsilon)
            child.fit(sub_features, sub_labels)
            self.child_dict[value] = child

    def __str__(self):
        node_information = f"node's idx_feature={self.idx_feature}\n" \
            f"node's child_dict={self.child_dict}\n" \
            f"node's label={self.label}\n"
        return node_information


class ClassifyDecisionTree(_Node):
    """
    分类决策树
    """

    def predict(self, feature):
        """
        预测数据
        :param feature: 特征
        :return: 预测的结果
        """
        print('*' * 10, '预测正在进行', '*' * 10)
        node = self
        while node.label is None:  # 注意不能用while not node.label，因为label可能为0
            to_delete_idx = node.idx_feature
            node = node.child_dict[feature[node.idx_feature]]
            feature.pop(to_delete_idx)
        return node.label


if __name__ == "__main__":
    # 《统计学习方法》的贷款申请样本数据表
    sample_with_labels = [
        [[0, 0, 0, 0], 0],
        [[0, 0, 0, 1], 0],
        [[0, 1, 0, 1], 1],
        [[0, 1, 1, 0], 1],
        [[0, 0, 0, 0], 0],
        [[1, 0, 0, 0], 0],
        [[1, 0, 0, 1], 0],
        [[1, 1, 1, 1], 1],
        [[1, 0, 1, 2], 1],
        [[1, 0, 1, 2], 1],
        [[2, 0, 1, 2], 1],
        [[2, 0, 1, 1], 1],
        [[2, 1, 0, 1], 1],
        [[2, 1, 0, 2], 1],
        [[2, 0, 0, 0], 0],
    ]
    test_features = [i[0] for i in sample_with_labels]
    test_labels = [i[1] for i in sample_with_labels]
    cdt = ClassifyDecisionTree(epsilon=0.1)
    cdt.fit(test_features, test_labels)
    print(cdt.predict([0, 1, 0, 0]))

    """
    用队列来先序遍历决策树的节点，打印出来
    方便按照打印信息来验证自己的树
    """
    q = deque([cdt])
    while q:
        if q[0].label:
            print(q.popleft())
        else:
            q.extend(q[0].child_dict.values())
            print(q.popleft())
