import numpy as np


# 求变换矩阵R的前n大的特征向量
def get_feature_vector(r, n, d):
    # r 变换矩阵 n 取前n个特征向量 d 样本维数
    value, vector = np.linalg.eig(r)

    # 特征值降序排列并保存特征值降序索引
    # sort_value = -np.sort(-value)
    sort_index = np.argsort(-value)
    # 按特征值索引取前n个特征向量 即最大的n个特征向量
    v = []
    for i in range(n):
        v.append(vector[:, sort_index[i]].reshape(d, 1))
    selected_vector = np.concatenate(v, axis=1)
    # selected_vector = np.concatenate((vector[:, sort_index[0]].reshape(100, 1),
    #                                   vector[:, sort_index[1]].reshape(100, 1)),
    #                                  axis=1)
    return selected_vector


# 求经过特征提取降维后的新特征样本
def get_new_sample(r, x, n, d):
    # r 变换矩阵 x 样本矩阵 n 取前n个特征向量 d 样本维数
    selected_vector = get_feature_vector(r, n, d)
    new_sample = np.dot(selected_vector.T, x)
    return new_sample
