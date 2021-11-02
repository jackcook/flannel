import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex

X = np.random.randn(1000, 2)

f = 2
t = AnnoyIndex(f, "euclidean")

for i in range(X.shape[0]):
    t.add_item(i, X[i])

t.build(10, n_jobs=1)

print(t.get_nns_by_item(0, 10, include_distances=True))
