#!/usr/bin/env python3
"""Figure 1 data: one-shot error of CLIP image embeddings vs number of leading principal components.
Input : embeddings_cifar100.npz   Output: images_error_vs_dim.json"""
import argparse, json, numpy as np
from common import pca_whiten, oneshot_error, h_alpha, fisher
ap = argparse.ArgumentParser(); ap.add_argument('--npz', default='embeddings_cifar100.npz'); ap.add_argument('--out', default='images_error_vs_dim.json')
ap.add_argument('--anchors', type=int, default=2500); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed)
d = np.load(a.npz, allow_pickle=True); X = d['X'].astype(np.float64); N = len(X)
P, lam = pca_whiten(X); cond = lam[0] / lam
n10, n100 = int(np.argmax(cond > 10)) + 1, int(np.argmax(cond > 100)) + 1
anchors = rng.choice(N, a.anchors, replace=False)
dims = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150, 200, 250]
rows = []
for n in dims:
    Z = P[:, :n]
    rows.append(dict(n=n, univ=oneshot_error(Z, anchors, h_alpha), fisher=oneshot_error(Z, anchors, fisher),
                     gauss=oneshot_error(rng.standard_normal((N, n)), anchors, h_alpha),
                     laplace=oneshot_error(rng.laplace(0, 1 / np.sqrt(2), (N, n)), anchors, h_alpha), cond=float(cond[n - 1])))
    print(f"n={n:3d} univ {rows[-1]['univ']:.2e} fisher {rows[-1]['fisher']:.2e} gauss {rows[-1]['gauss']:.2e} laplace {rows[-1]['laplace']:.2e} cond {cond[n-1]:.1f}", flush=True)
# full dimension, all anchors, violating ordered pairs
Z = P; viol = 0
for s in range(0, N, 1000):
    A = Z[s:s + 1000]; Cn = h_alpha(A); thr = np.einsum('bn,bn->b', A, Cn); V = (Z @ Cn.T) >= thr[None, :]
    V[np.arange(s, s + len(A)), np.arange(len(A))] = False; viol += int(V.sum())
ns = np.array([r['n'] for r in rows]); eu = np.array([r['univ'] for r in rows]); m = (ns <= 60) & (eu > 0)
slope, inter = np.polyfit(ns[m], np.log(eu[m]), 1)
json.dump(dict(rows=rows, n_cond10=n10, n_cond100=n100, fit_factor=float(np.exp(slope)), fit_intercept=float(inter), fit_range='n<=60',
               full_dim_violating_ordered_pairs=viol, resolution=1 / (a.anchors * (N - 1)), eig=dict(l1=float(lam[0]), l64=float(lam[63]), l256=float(lam[255]), l448=float(lam[447]), l512=float(lam[511]))),
          open(a.out, 'w'), indent=1)
print(f'condition number > 10 from n={n10}, > 100 from n={n100}; fit factor {np.exp(slope):.3f}; full-dim violating pairs {viol}')
