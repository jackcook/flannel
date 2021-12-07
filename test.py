import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from flannel import AnnoyIndex as FlannelIndex
from tqdm import trange, tqdm
import sys
import os
import pickle

if not os.path.exists("glove_vecs.npz"):
    df = pd.read_csv("unigram_freq.csv")
    freqs = {x.word: x.count for x in tqdm(df.itertuples())}

    def load_glove_model(fname):
        vecs = []
        weights = []

        with open(fname, 'r') as f:
            for line in tqdm(f):
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)

                if word not in freqs:
                    continue
                
                vecs.append(embedding)
                weights.append(freqs[word])

        return vecs, weights

    vecs, weights = load_glove_model("glove.twitter.27B/glove.twitter.27B.25d.txt")
    np.savez("glove_vecs.npz", vecs=vecs, weights=weights)

f = np.load("glove_vecs.npz")
vecs = f["vecs"]
weights = f["weights"]

f = 25
# n_items = 10000
# vecs = np.random.randn(n_items, f)
# weights = np.random.zipf(2, size=len(vecs))

n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 10
n_trees = 10
search_k = 100
n_items = vecs.shape[0]
k = 10

old_scores = []
new_scores = []

t_old = AnnoyIndex(f, "angular")
t = FlannelIndex(f, "angular")

p = weights / weights.sum()

if not os.path.exists("test_old.ann"):
    for i in range(n_items):
        t_old.add_item(i, vecs[i])
    
    t_old.build(n_trees)
    t_old.save("test_old.ann")

if not os.path.exists("test.ann"):
    for i in range(n_items):
        t.add_item(i, vecs[i], p[i])

    t.build(n_trees, 100, top_p=0.05, with_neighbors=True, n_neighbors=k)
    t.save("test.ann")

t_old.load("test_old.ann")
t.load("test.ann")

def get_gt_idx(item_i):
    if item_i in nearest:
        return nearest[item_i]

    gt_idx = t_old.get_nns_by_item(item_i, k, search_k=n_items * n_trees)
    nearest[item_i] = gt_idx

    return gt_idx

nearest = {}

if os.path.exists("nearest.pkl"):
    nearest = pickle.load(open("nearest.pkl", "rb"))

# item_i = 3
# new_idx = t.get_nns_by_item(item_i, k, search_k=search_k, clusters_p=0.00)

# print(get_gt_idx(item_i))
# print(new_idx)
# assert 0

it = range(n_iter)

if n_iter != 1:
    it = trange(n_iter)

for a in it:
    np.random.seed(a)

    for _ in range(100):
        item_i = np.random.choice(range(n_items), p=p / p.sum())
        gt_idx = get_gt_idx(item_i)

        old_idx = t_old.get_nns_by_item(item_i, k, search_k=search_k)
        old_matches = len(set(gt_idx).intersection(set(old_idx)))
        old_score = old_matches / k
        assert item_i in old_idx

        new_idx = t.get_nns_by_item(item_i, k, search_k=search_k)
        new_matches = len(set(gt_idx).intersection(set(new_idx)))
        new_score = new_matches / k
        assert item_i in new_idx

        old_scores.append(old_score)
        new_scores.append(new_score)

print(f"Old score: {np.mean(old_scores)}")
print(f"New score: {np.mean(new_scores)}")

with open("nearest.pkl", "wb") as f:
    pickle.dump(nearest, f)