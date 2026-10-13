#!/usr/bin/env python3
"""Figure 2 (right) data: capacity of the label space.  For each n, the largest m such that m random labels
(cleaned vocabulary V2, n leading whitened principal components) are all one-shot separable with probability >= 1-delta,
found by bisection with Monte-Carlo trials, against the bound sqrt(delta / c_hat) of Proposition 3.1.
Input : labels_prepared.npz   Output: labels_capacity.json"""
import argparse, json, time, numpy as np
from common import oneshot_error, h_alpha, fisher
ap = argparse.ArgumentParser(); ap.add_argument('--prep', default='labels_prepared.npz'); ap.add_argument('--out', default='labels_capacity.json')
ap.add_argument('--dims', default='10,20,30,40,50,60,80,100,150,200,250'); ap.add_argument('--delta', type=float, default=0.1)
ap.add_argument('--trials', type=int, default=200); ap.add_argument('--seed', type=int, default=5); a = ap.parse_args()
rng = np.random.default_rng(a.seed); d = np.load(a.prep, allow_pickle=True); Z = d['P'][d['idx_V2']]; M = len(Z); t0 = time.time()
maps = {'univ': h_alpha, 'fisher': fisher}
def all_sep(Zs, cmap):
    Cn = cmap(Zs); thr = np.einsum('bn,bn->b', Zs, Cn); S = Zs @ Cn.T - thr[None, :]; np.fill_diagonal(S, -1.0); return bool((S < 0).all())
def capacity(Zn, cmap, lo, hi):
    ok = lambda m: np.mean([all_sep(Zn[rng.choice(len(Zn), m, replace=False)], cmap) for _ in range(a.trials)]) >= 1 - a.delta
    if not ok(lo): return lo
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if ok(mid): lo = mid
        else: hi = mid
    return lo
out = {}
for n in [int(x) for x in a.dims.split(',')]:
    Zn = Z[:, :n]; row = {}
    for name, cmap in maps.items():
        anchors = rng.choice(M, 1000, replace=False); c = oneshot_error(Zn, anchors, cmap); pred = float(np.sqrt(a.delta / c)) if c > 0 else float('inf')
        pe = min(pred, 4000.0); m = capacity(Zn, cmap, lo=max(2, int(pe / 5)), hi=min(int(pe * 5) + 10, 4000, M)); row[name] = dict(capacity=int(m), pred=pred, c_hat=float(c))
    out[n] = row; print(f"n={n:3d} univ m*={row['univ']['capacity']:4d} (bound {row['univ']['pred']:6.0f}) fisher m*={row['fisher']['capacity']:4d}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(delta=a.delta, trials=a.trials, vocabulary='V2', results=out), open(a.out, 'w'), indent=1)
