import numpy as np
from annoy import AnnoyIndex
from flannel import AnnoyIndex as FlannelIndex
from tqdm import trange
import sys

n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 10
f = 25
n_items = 5000
n_trees = 10
search_k = 200

old_scores = []
new_scores = []

it = range(n_iter)

if n_iter != 1:
    it = trange(n_iter)

for a in it:
    np.random.seed(a)
    X = np.random.randn(n_items, f)

    t_old = AnnoyIndex(f, "angular")

    for i in range(n_items):
        t_old.add_item(i, X[i])
    
    t_old.build(n_trees)

    t = FlannelIndex(f, "angular")

    p = np.random.exponential(1.5, size=(n_items,))

    for i in range(n_items):
        w = p[i]
        neighbors = []

        if w > np.percentile(p, 98):
            neighbors = t_old.get_nns_by_item(i, 10, search_k=n_items * n_trees)

        t.add_item(i, X[i], w, neighbors=neighbors)

    t.build(n_trees, 50, top_p=0.01, with_neighbors=True)
    t.save("test.ann")

    t = FlannelIndex(f, "angular")
    t.load("test.ann")

    item_i = np.argmax(p)

    k = 10
    gt_idx, gt_dist = t_old.get_nns_by_item(item_i, k, search_k=n_items * 2, include_distances=True)

    old_idx, old_dist = t_old.get_nns_by_item(item_i, k, search_k=search_k, include_distances=True)
    old_matches = len(set(gt_idx).intersection(set(old_idx)))
    old_score = old_matches / k
    assert item_i in old_idx

    new_idx, new_dist = t.get_nns_by_item(item_i, k, search_k=search_k, clusters_p=0.5, include_distances=True)
    new_matches = len(set(gt_idx).intersection(set(new_idx)))
    new_score = new_matches / k
    assert item_i in new_idx

    old_scores.append(old_score)
    new_scores.append(new_score)

print(f"Old score: {np.mean(old_scores)}")
print(f"New score: {np.mean(new_scores)}")
