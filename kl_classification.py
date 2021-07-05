import numpy as np
import matplotlib.pyplot as plt
from function import get_new_sample

# 将两类100维样本降到2维
# 每类50个数据
# 用K-L变换算法

# 样本数、类数、维数
size, num, dimension = 50, 2, 100

# 两类服从正态分布的样本 100 * 50 -> 100 * 2
# 第一类样本 均值0 方差1
c1 = np.random.normal(0, 1, [dimension, size])
# 第二类样本 均值10 方差5
c2 = np.random.normal(10, 5, [dimension, size])
# 将两类样本联立成一个矩阵 100 * 100
x = np.concatenate((c1, c2), axis=1)

# 求变换矩阵R
# X * X.T = x1 * x1.T + x2 * x2.T + ...
# R = E[x * x.T] = X * X.T / dimension
z = np.dot(x, x.T)
R = z / (size * 2)
# 获得新特征值的样本（变换矩阵R）
s1 = get_new_sample(R, x, 2, dimension)

# 求协方差矩阵
R_R = np.cov(x, bias=True)
# 获得新特征值的样本（协方差矩阵）
s2 = get_new_sample(R_R, x, 2, dimension)

# 画点
# 用R = E[x * x.T]矩阵分类后的结果
plt.plot(s1[0], s1[1], '.')
# 用协方差矩阵分类后的结果
plt.plot(s2[0], s2[1], '+')

plt.show()

print(R)
print(R_R)
