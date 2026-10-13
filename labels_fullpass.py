#!/usr/bin/env python3
"""Full-dimensional one-shot separability of the label embeddings (sphere-projected, centred; no whitening):
all ordered pairs, universal map and Fisher map, for the full vocabulary and the cleaned vocabularies V1, V2.
Reports the number of violating pairs, the labels affected and the most/least similar violating pairs.
Input : labels_prepared.npz   Output: labels_fullpass.json   (about 1-5 min per vocabulary on one core)"""
import argparse, json, time, numpy as np
from common import h_alpha, fisher
ap = argparse.ArgumentParser(); ap.add_argument('--prep', default='labels_prepared.npz'); ap.add_argument('--out', default='labels_fullpass.json')
ap.add_argument('--vocab', default='all,V1,V2'); ap.add_argument('--maps', default='univ,fisher'); a = ap.parse_args()
d = np.load(a.prep, allow_pickle=True); Zc, names = d['Zc'], d['names']; maps = {'univ': h_alpha, 'fisher': fisher}
try: out = json.load(open(a.out))
except Exception: out = {}
for vocab in a.vocab.split(','):
    idx = np.arange(len(Zc)) if vocab == 'all' else d['idx_' + vocab]; Z = Zc[idx]; Z = (Z - Z.mean(0)).astype(np.float32); nm = names[idx]; M = len(Z)
    for mp in a.maps.split(','):
        t0 = time.time(); cmap = maps[mp]; Cn = cmap(Z); thr = np.einsum('bn,bn->b', Z, Cn); pairs = []; affected = np.zeros(M, bool)
        for s in range(0, M, 512):
            S = Z @ Cn[s:s + 512].T; S -= thr[None, s:s + 512]; b = S.shape[1]; S[np.arange(s, s + b), np.arange(b)] = -1.0
            jj, ii = np.nonzero(S >= 0); pairs += [(int(s + i), int(j)) for i, j in zip(ii, jj)]; affected[s:s + b] |= (S >= 0).any(0); del S
        un = {tuple(sorted(p)) for p in pairs}; Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        ex = sorted([(str(nm[i]), str(nm[j]), float(Zn[i] @ Zn[j])) for i, j in un], key=lambda t: -t[2])
        out[f'{vocab}|{mp}'] = dict(M=M, violating_unordered_pairs=len(un), labels_affected=int(affected.sum()), fraction_affected=float(affected.mean()),
                                   most_similar=ex[:30], least_similar=ex[-15:])
        print(f'{vocab:3s} {mp:6s}: M={M} violating pairs {len(un)} labels affected {affected.sum()} ({100*affected.mean():.2f}%) [{time.time()-t0:.0f}s]', flush=True)
        json.dump(out, open(a.out, 'w'), indent=1)
