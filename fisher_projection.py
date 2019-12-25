import numpy as np
import matplotlib.pyplot as plt

# 在二维平面上生成2类分布点
# 每类50个数据
# 用Fisher判别准则找投影面 使类内最紧密 类间最分散

# 样本数
size = 50
# 维数
dimension = 2

# 第一类样本 均值0 方差1
c1 = np.random.normal(0, 1, [dimension, size])
# 第二类样本 均值20 方差5
c2 = np.random.normal(20, 5, [dimension, size])
# 将两类样本联立成一个矩阵 2 * 100
x = np.concatenate((c1, c2), axis=1)

# sum(axis=1) 每行的所有元素相加
# 各类均值向量
m1 = c1.sum(axis=1) / size  # (2, )
m2 = c2.sum(axis=1) / size  # (2, )
# 增加维度
# m1 = np.expand_dims(m1, axis=1)  # (2, 1)
# m2 = np.expand_dims(m2, axis=1)  # (2, 1)
m1 = m1.reshape(2, 1)
m2 = m2.reshape(2, 1)
# 各类的类内离散矩阵
# (2, 50) - (2, 1) = (2, 50)
s1 = np.dot((c1 - m1), (c1 - m1).T)  # (2, 50) * (50 * 2) = (2, 2)
s2 = np.dot((c2 - m2), (c2 - m2).T)  # (2, 50) * (50 * 2) = (2, 2)
# 总类内离散矩阵 sw
sw = s1 + s2  # (2, 2)
print(sw)
sw_inv = np.linalg.inv(sw)  # (2, 2)

# 投影向量
w_ = np.dot(sw_inv, (m1 - m2))  # (2, 2) * (2, 1) = (2, 1)

print(w_)
plt.plot(x[0], x[1], '.')
k = w_[1] / w_[0]
plt.plot([0, 30], [0, k * 30])
plt.show()
