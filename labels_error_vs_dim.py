#!/usr/bin/env python3
"""Figure 2 (left) data: one-shot error of label embeddings vs number of leading principal components,
for the WordNet vocabulary (all distinct lemmas and the cleaned vocabulary V2) and the ImageNet-1k class names.
Input : labels_prepared.npz   Output: labels_error_vs_dim.json"""
import argparse, json, numpy as np
from common import oneshot_error, h_alpha
ap = argparse.ArgumentParser(); ap.add_argument('--prep', default='labels_prepared.npz'); ap.add_argument('--out', default='labels_error_vs_dim.json')
ap.add_argument('--anchors', type=int, default=1000); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed); d = np.load(a.prep, allow_pickle=True)
P, lam, idx2, Pin, lam_in = d['P'], d['lam'], d['idx_V2'], d['P_in'], d['lam_in']
dims = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150, 200, 250]
sets = {'wordnet_all': P, 'wordnet_V2': P[idx2], 'imagenet1k': Pin}
out = {k: [] for k in sets}
for name, Z in sets.items():
    anchors = np.arange(len(Z)) if len(Z) <= 2000 else rng.choice(len(Z), a.anchors, replace=False)
    lamk = lam_in if name == 'imagenet1k' else lam
    for n in dims:
        e = oneshot_error(Z[:, :n], anchors, h_alpha); out[name].append(dict(n=n, err=e, cond=float(lamk[0] / lamk[n - 1])))
        print(f'{name:12s} n={n:3d} err {e:.2e} cond {lamk[0]/lamk[n-1]:.1f}', flush=True)
json.dump(dict(results=out, resolution_wordnet=1 / (a.anchors * (len(P) - 1)), resolution_imagenet=1 / (1000 * 999),
               cond_thresholds=dict(wordnet_n10=int(np.argmax(lam[0] / lam > 10)) + 1, wordnet_n100=int(np.argmax(lam[0] / lam > 100)) + 1)),
          open(a.out, 'w'), indent=1)
