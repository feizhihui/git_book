# encoding=utf-8
# from sklearn.decomposition import NMF
# import numpy as np
#
# # V=WH
# origin_W = np.array([[1, 2, 3], [4, 5, 6]])  # 2*3
# origin_H = np.array([[1, 2], [3, 4], [5, 6]])  # 3*2
# V = np.dot(origin_W, origin_H)
#
# model = NMF(n_components=5)
# new_W = model.fit_transform(V)
# # V=WH
# new_V = np.dot(new_W, model.components_)
# print('===============')
# print(new_V)
# print(V)
# print('===============', new_W.shape, model.components_.shape)

import numpy as np

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF

model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

X_new = np.array([[1, 0], [1, 6.1], [1, 0], [1, 4], [3.2, 1], [0, 4]])
W_new = model.transform(X_new)

# print(W_new)
# print(X_new @ np.linalg.inv(H))

print(W_new @ model.components_)
