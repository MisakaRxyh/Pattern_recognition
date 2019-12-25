import numpy as np
import matplotlib.pyplot as plt
from function import get_feature_vector

# 在二维平面上生成3类分布点
# 每类100个数据
# 用K-L变换找投影面

# 样本数
size = 100
# 维数
dimension = 2
# 样本点 normal 正态 exponential 指数 poisson 泊松
c1 = np.random.normal(5, 1, [dimension, size])
# c1 = np.random.exponential(5, [dimension, size])
c2 = np.random.normal(10, 2, [dimension, size])
# c2 = np.random.poisson(1, [dimension, size])
c3 = np.random.normal(20, 4, [dimension, size])

x = np.concatenate((c1, c2, c3), axis=1)

R_R = np.cov(x)
sv = get_feature_vector(R_R, 1, dimension)

plt.plot(x[0], x[1], '.')
k = sv[1] / sv[0]
plt.plot([0, 30], [0, k * 30])
plt.show()
