import numpy as np
import matplotlib.pyplot as plt
from function import get_feature_vector

# 在二维平面上生成3类分布点
# 每类100个数据
# 用K-L变换找投影面

# 样本数、类数、维数
size, num, dimension = 100, 3, 2

# 样本点 normal 正态 exponential 指数 poisson 泊松
c1 = np.random.normal(5, 1, [dimension, size])
# c1 = np.random.exponential(5, [dimension, size])
c2 = np.random.normal(10, 2, [dimension, size])
# c2 = np.random.poisson(1, [dimension, size])
c3 = np.random.normal(20, 4, [dimension, size])

# 更离散的点
# c1x = c21 = np.random.normal(0, 3, [1, size])
# c1y = np.random.normal(30, 1, [1, size])
# c1 = np.concatenate((c1x, c1y), axis=0)
#
# c2x = np.random.normal(5, 3, [1, size])
# c2y = np.random.normal(15, 1, [1, size])
# c2 = np.concatenate((c2x, c2y), axis=0)
#
# c3x = np.random.normal(25, 3, [1, size])
# c3y = np.random.normal(10, 1, [1, size])
# c3 = np.concatenate((c3x, c3y), axis=0)

# 将三类分布的点合并成一个矩阵
x = np.concatenate((c1, c2, c3), axis=1)
# 计算协方差矩阵
R_R = np.cov(x, bias=True)
# z = np.dot(x, x.T)
# R = z / (size * num)

# 获取协方差矩阵的最大向量
fv = get_feature_vector(R_R, 1, dimension)
# 画图
plt.plot(x[0], x[1], '.')
k = fv[1] / fv[0]
plt.plot([0, 30], [0, k * 30])
plt.show()
