#!/usr/bin/env python3
"""Deduplicate the WordNet label embeddings, build the cleaned vocabularies and the PCA-whitened scores.
Input : text_embeddings.npz (T_wordnet, names_wordnet, T_imagenet, names_imagenet)
Output: labels_prepared.npz with
        names        distinct WordNet lemmas (after removing identical embeddings)
        Zc           sphere-projected, centred embeddings            [M,512]
        P, lam       PCA-whitened scores and covariance eigenvalues   [M,512], [512]
        idx_V1       indices of one lemma per WordNet synset (concept vocabulary)
        idx_V2       idx_V1 without plant/animal taxa
        Zc_in, P_in, lam_in, names_in   the same for the ImageNet-1k class names
"""
import argparse, numpy as np
from common import pca_whiten, sphere_centre

ap = argparse.ArgumentParser(); ap.add_argument('--npz', default='text_embeddings.npz'); ap.add_argument('--out', default='labels_prepared.npz')
a = ap.parse_args()
d = np.load(a.npz, allow_pickle=True)
Tw = d['T_wordnet'].astype(np.float32); names = np.array(d['names_wordnet']); Ti = d['T_imagenet'].astype(np.float32); inames = np.array(d['names_imagenet'])
key = np.ascontiguousarray(np.round(Tw, 4)).view(np.dtype((np.void, Tw.shape[1] * 4))).ravel()
_, keep = np.unique(key, return_index=True); keep = np.sort(keep)
Tw, names = Tw[keep], names[keep]
print(f'WordNet lemmas {len(d["names_wordnet"])}, distinct embeddings {len(Tw)}')
import nltk; nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn
S = {str(n): i for i, n in enumerate(names)}; V1, V2 = set(), set()
for s in wn.all_synsets('n'):
    lem = [l.name() for l in s.lemmas() if l.name() in S]
    if not lem: continue
    V1.add(lem[0])
    if s.lexname() not in ('noun.plant', 'noun.animal'): V2.add(lem[0])
idx_V1 = np.array(sorted(S[w] for w in V1)); idx_V2 = np.array(sorted(S[w] for w in V2))
print(f'concept vocabulary V1: {len(idx_V1)}; without plants/animals V2: {len(idx_V2)}')
P, lam = pca_whiten(Tw); P_in, lam_in = pca_whiten(Ti)
np.savez_compressed(a.out, names=names, Zc=sphere_centre(Tw), P=P, lam=lam, idx_V1=idx_V1, idx_V2=idx_V2,
                    names_in=inames, Zc_in=sphere_centre(Ti), P_in=P_in, lam_in=lam_in)
print('saved', a.out)
