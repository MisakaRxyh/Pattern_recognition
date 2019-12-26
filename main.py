import numpy as np
import matplotlib.pyplot as plt
from function import get_feature_vector
# 样本数
size = 100
# 维数
dimension = 2

# c1 = np.random.normal(0, 1, [dimension, size])
# c2 = np.random.normal(10, 5, [dimension, size])
# c3 = np.random.normal(20, 3, [dimension, size])

c1x = c21 = np.random.normal(0, 3, [1, size])
c1y = np.random.normal(30, 1, [1, size])
c1 = np.concatenate((c1x, c1y), axis=0)

c2x = np.random.normal(5, 3, [1, size])
c2y = np.random.normal(15, 1, [1, size])
c2 = np.concatenate((c2x, c2y), axis=0)

c3x = np.random.normal(25, 3, [1, size])
c3y = np.random.normal(10, 1, [1, size])
c3 = np.concatenate((c3x, c3y), axis=0)

# 求变换矩阵R
x = np.concatenate((c1, c2, c3), axis=1)
z = np.dot(x, x.T)
R = z / (size * 3)
R_R = np.cov(x, bias=True)
# 求变换矩阵R特征值与特征向量
value, vector = np.linalg.eig(R)
sort_value = -np.sort(-value)
sort_index = np.argsort(-value)
Selected_vector = get_feature_vector(R_R, 1, dimension)
print(Selected_vector)
# 画点
plt.plot(x[0], x[1], '.')
k = Selected_vector[1] / Selected_vector[0]
plt.plot([0, 30], [0, k * 30])

plt.show()
