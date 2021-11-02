import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex

f = 10

np.random.seed(0)
X = np.random.randn(10000, f)

t = AnnoyIndex(f, "euclidean")

for i in range(X.shape[0]):
    t.add_item(i, X[i])

print("building")
t.build(10, n_jobs=1)

print(t.get_nns_by_item(0, 10, include_distances=True))
print("done")
