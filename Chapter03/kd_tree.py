# coding: utf-8
import math
from heapq import heappop, heappush, nsmallest


def find_middle(X, Y, dim):
    """
    找到dim纬度上处于中位数的实例，返回这个实例和更小、更大的X,Y
    :param X:
    :param Y:
    :param dim:
    :return:
    """
    # print(X, Y)
    sorted_X_Y = sorted(zip(X, Y), key=lambda x_and_y: x_and_y[0][dim])
    middle_index = len(X) >> 1
    middle = sorted_X_Y[middle_index]

    smaller = sorted_X_Y[:middle_index]
    bigger = sorted_X_Y[middle_index + 1:]

    smaller_X, smaller_Y = [i[0] for i in smaller], [i[1] for i in smaller]
    bigger_X, bigger_Y = [i[0] for i in bigger], [i[1] for i in bigger]
    smaller_X, smaller_Y, bigger_X, bigger_Y = list(smaller_X), list(smaller_Y), list(bigger_X), list(bigger_Y)
    return middle, smaller_X, smaller_Y, bigger_X, bigger_Y


def l2(x1, x2):
    return math.sqrt(sum([(x_1_i - x_2_i) ** 2 for x_1_i, x_2_i in zip(x1, x2)]))


class Node:
    """Node的实例代表KDTree的一个节点"""

    def __repr__(self):
        return f"深度为{self.level}, 以第{self.dim}个特征作为分割标准, 实例点为{self.instance}"

    def __init__(self, instance, level=0):
        self.instance = instance
        self.level = level
        self.left = None
        self.right = None
        self.parent = None

    @property
    def dim(self):
        return self.level % len(self.instance)

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def brother(self):
        if self.parent is None:
            return None
        if self.parent.left is self:  # 当自己是父节点的左子节点，则兄弟节点为父节点的右节点
            return self.parent.right
        return self.parent.left  # 反之

    def plane_distance(self, x):
        """节点所代表的超平面与目标点的距离"""
        return abs(x[self.dim] - self.instance[0][self.dim])

    def point_distance(self, x):
        return l2(self.instance[0], x)

    def find_leaf(self, x):
        node = self
        while not node.is_leaf:
            if node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            elif x[node.dim] < node.instance[0][node.dim]:
                node = node.left
            else:
                node = node.right
        return node


class KDTree:
    def __repr__(self):
        representation = ""
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            representation += str(node)
            representation += '\n'
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return representation

    def __init__(self, X, Y):
        def _build_node(_X, _Y, _level, _dim):
            """递归地方式构建节点"""
            _middle, _smaller_X, _smaller_Y, _bigger_X, _bigger_Y = find_middle(_X, _Y, _dim)
            # print(_middle, _smaller_X, _smaller_Y, _bigger_X, _bigger_Y)
            _node = Node(_middle, _level)
            _next_level = _level + 1
            _next_dim = _next_level % len(_middle)
            if _smaller_X:
                _node.left = _build_node(_smaller_X, _smaller_Y, _next_level, _next_dim)
            if _bigger_X:
                _node.right = _build_node(_bigger_X, _bigger_Y, _next_level, _next_dim)
            return _node

        self.root = _build_node(X, Y, 0, 0)
        # 递归设置父节点
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.left:
                node.left.parent = node
                queue.append(node.left)
            if node.right:
                node.right.parent = node
                queue.append(node.right)

    def search(self, x, k):
        """找到最接近x的k个实例"""

        def backtrack(root, knn_list, is_visited):
            if root is self.root and root in is_visited:
                return

            node = root.find_leaf(x)
            is_visited.append(node)
            dist = node.point_distance(x)

            if len(knn_list) < k:
                # record = (-距离, 实例点),heappush构造的是小顶堆，而我们想知道的是最大距离点，故对距离取相反数
                heappush(knn_list, (-dist, node.instance))
            else:
                # 先比较这个叶节点是否比knn_list中最远点近，是的话替换，否则不换
                farthest_dist, farthest_point = nsmallest(1, knn_list)[0]
                if -farthest_dist > dist:
                    heappop(knn_list)
                    heappush(knn_list, (-dist, node.instance))

            # 往上寻找没有被访问过的父节点，并将兄弟节点取出备用
            brother = node.brother
            node = node.parent
            while node in is_visited and node.parent:
                brother = node.brother
                node = node.parent
            # 如果遍历到顶
            if node is self.root and node in is_visited:
                return

            while True:
                # 否则计算父节点是否能满足条件、并把父节点计入被访问列表
                is_visited.append(node)
                dist = node.point_distance(x)
                if len(knn_list) < k:
                    # record = (距离, 实例点)
                    # heappush构造的是小顶堆，而我们想知道的是最大距离点，故对距离取相反数
                    heappush(knn_list, (-dist, node.instance))
                else:
                    # 先比较这个叶节点是否比knn_list中最远点近，是的话替换，否则不换
                    farthest_dist, farthest_point = nsmallest(1, knn_list)[0]
                    if -farthest_dist > dist:
                        heappop(knn_list)
                        heappush(knn_list, (-dist, node.instance))

                # 再看超平面
                farthest_dist, farthest_point = nsmallest(1, knn_list)[0]
                if (node.plane_distance(x) < -farthest_dist or len(knn_list) < k) and brother is not None:
                    backtrack(brother, knn_list, is_visited)
                    break
                else:
                    while node in is_visited and node.parent:
                        brother = node.brother
                        node = node.parent
                    # 如果遍历到顶
                    if node is self.root and node in is_visited:
                        return

        _knn_list = []
        _is_visited = []
        backtrack(self.root, _knn_list, _is_visited)
        print(_knn_list)
        return _knn_list
