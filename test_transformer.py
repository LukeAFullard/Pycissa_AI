import numpy as np
from pycissa.preprocessing import MultiplicativeTransformer

T = 100
t = np.arange(T)
X_1d = np.sin(t) * 5.0 - 2.0

transformer = MultiplicativeTransformer()
X_log = transformer.fit_transform(X_1d)
print(transformer.offsets)

X_rec = transformer.inverse_transform(X_log)
print("1D Match:", np.allclose(X_1d, X_rec))

X_2d = np.column_stack([np.sin(t)*5 - 2, np.cos(t)*10 + 20])
trans2 = MultiplicativeTransformer()
X_2d_log = trans2.fit_transform(X_2d, columns_to_transform=[0, 1])
print(trans2.offsets)

X_2d_rec_0 = trans2.inverse_transform(X_2d_log[:, 0], col_idx=0)
X_2d_rec_1 = trans2.inverse_transform(X_2d_log[:, 1], col_idx=1)

print("2D Match 0:", np.allclose(X_2d[:, 0], X_2d_rec_0))
print("2D Match 1:", np.allclose(X_2d[:, 1], X_2d_rec_1))
