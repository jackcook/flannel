import numpy as np
from annoy import AnnoyIndex
from flannel import AnnoyIndex as FlannelIndex
from tqdm import trange

f = 100

old_scores = []
new_scores = []

for a in trange(10):
    np.random.seed(a)
    X = np.random.randn(10000, f)

    t_old = AnnoyIndex(f, "angular")
    t = FlannelIndex(f, "angular")

    for i in range(X.shape[0]):
        t_old.add_item(i, X[i])
        t.add_item(i, X[i], 3)

    # print("building")
    t_old.build(10, n_jobs=1)
    t.build(10, 10, n_jobs=1)

    k = 10
    gt_idx, gt_dist = t_old.get_nns_by_item(0, k, search_k=1000000, include_distances=True)

    old_idx, old_dist = t_old.get_nns_by_item(0, k, search_k=2000, include_distances=True)
    old_matches = len(set(gt_idx).intersection(set(old_idx)))
    old_score = old_matches / k
    # print(f"Old score: {old_score}")

    new_idx, new_dist = t.get_nns_by_item(0, k, search_k=1000, include_distances=True)
    new_matches = len(set(gt_idx).intersection(set(new_idx)))
    new_score = new_matches / k
    # print(f"New score: {new_score}")

    old_scores.append(old_score)
    new_scores.append(new_score)

print(f"Old score: {np.mean(old_scores)}")
print(f"New score: {np.mean(new_scores)}")
