# encoding=utf-8
from sklearn.decomposition import NMF
import numpy as np

# V=WH
origin_W = np.array([[1, 2, 3], [4, 5, 6]])  # 2*3
origin_H = np.array([[1, 2], [3, 4], [5, 6]])  # 3*2
V = np.dot(origin_W, origin_H)

model = NMF(n_components=5)
new_W = model.fit_transform(V)
# V=WH
new_V = np.dot(new_W, model.components_)
print('===============')
print(new_V)
print(V)
print('===============', new_W.shape, model.components_.shape)
