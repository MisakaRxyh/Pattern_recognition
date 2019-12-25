import numpy as np
import matplotlib.pyplot as plt

# 样本数
size = 100
# 维数
dimension = 2

c1 = np.random.normal(0, 1, [dimension, size])
c2 = np.random.normal(10, 5, [dimension, size])
c3 = np.random.normal(20, 3, [dimension, size])
# 求变换矩阵R
x = np.concatenate((c1, c2, c3), axis=1)
z = np.dot(x, x.T)
R = z / (size * 3)
R_R = np.cov(x, bias=True)
# 求变换矩阵R特征值与特征向量
value, vector = np.linalg.eig(R)
sort_value = -np.sort(-value)
sort_index = np.argsort(-value)
Selected_vector = np.concatenate((vector[:, sort_index[0]].reshape(2, 1)))
# 画点
plt.plot(x[0], x[1], '.')
k = Selected_vector[1] / Selected_vector[0]
plt.plot([0, 30], [0, k * 30])

plt.show()
