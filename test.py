import numpy as np
from annoy import AnnoyIndex
from flannel import AnnoyIndex as FlannelIndex
from tqdm import trange
import sys

n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 10
f = 100
n_items = 10000
search_k = 2000

old_scores = []
new_scores = []

it = range(n_iter)

if n_iter != 1:
    it = trange(n_iter)

for a in it:
    np.random.seed(a)
    X = np.random.randn(n_items, f)

    t_old = AnnoyIndex(f, "angular")
    t = FlannelIndex(f, "angular")

    for i in range(X.shape[0]):
        t_old.add_item(i, X[i])
        t.add_item(i, X[i], 3)

    # print("building")
    t_old.build(10, n_jobs=1)
    t.build(10, 5, n_jobs=1)
    t.save("test.ann")

    t = FlannelIndex(f, "angular")
    t.load("test.ann")

    k = 10
    gt_idx, gt_dist = t_old.get_nns_by_item(0, k, search_k=n_items * 2, include_distances=True)

    old_idx, old_dist = t_old.get_nns_by_item(0, k, search_k=search_k, include_distances=True)
    old_matches = len(set(gt_idx).intersection(set(old_idx)))
    old_score = old_matches / k
    assert 0 in old_idx

    new_idx, new_dist = t.get_nns_by_item(0, k, search_k=search_k // 2, include_distances=True)
    new_matches = len(set(gt_idx).intersection(set(new_idx)))
    new_score = new_matches / k
    assert 0 in new_idx

    old_scores.append(old_score)
    new_scores.append(new_score)

print(f"Old score: {np.mean(old_scores)}")
print(f"New score: {np.mean(new_scores)}")
