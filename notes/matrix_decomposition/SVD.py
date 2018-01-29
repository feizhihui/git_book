# encoding=utf-8


import numpy as np


def load_data():
    # 7*5
    return np.array([[0, 0, 0, 2, 2],
                     [0, 0, 0, 3, 3],
                     [0, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0],
                     [2, 2, 2, 0, 0],
                     [5, 5, 5, 0, 0],
                     [1, 1, 1, 0, 0]])


def eig_transform(data, sigma):
    a = np.eye(len(sigma), dtype=np.float32) * sigma
    b = np.zeros_like(data, dtype=np.float32)
    b[:len(sigma), :] += a
    return b


data = load_data()
u, sigma, v = np.linalg.svd(data)
s = eig_transform(data, sigma)
fit = np.dot(np.dot(u, s), v)
delta = fit - data
# print(np.abs(delta) < 1e-6)

print(sigma)

# 求解协方差矩阵特征值
data_square = np.dot(data, data.T)
eig, Q = np.linalg.eig(data_square)
# 实对称矩阵特征值为正
eig = np.round(eig, 9)
delta = eig_transform(data_square, eig)
# print(data_square)
# print(np.dot(np.dot(Q, delta), Q.T))
# print(Q @ delta @ Q.T)
# 奇异值的平方等于特征值
print(sorted(np.sqrt(eig), reverse=True))
