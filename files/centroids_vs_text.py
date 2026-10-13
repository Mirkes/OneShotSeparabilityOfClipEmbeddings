#!/usr/bin/env python3
"""Text embeddings of class names versus centroids of the image embeddings (CIFAR-100).
Reports the mutual nearest-neighbour matching, the common translation (modality gap), the class-specific residual,
and the accuracy of nearest-true-centroid versus text classification.   Input: embeddings_cifar100.npz   Output: centroids_vs_text.json"""
import argparse, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--npz', default='embeddings_cifar100.npz'); ap.add_argument('--out', default='centroids_vs_text.json'); a = ap.parse_args()
d = np.load(a.npz, allow_pickle=True); X = d['X'].astype(np.float64); y = d['y']; T = d['T'].astype(np.float64); C = len(T)
unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
mu = np.stack([X[y == k].mean(0) for k in range(C)]); gap = T - mu; g = gap.mean(0); res = gap - g
out = {}
for name, A, B in [('raw_inner_product', mu, T), ('cosine', unit(mu), unit(T)), ('cosine_each_modality_centred', unit(mu - mu.mean(0)), unit(T - T.mean(0)))]:
    S = A @ B.T; out[name] = dict(top1_centroid_to_text=float((S.argmax(1) == np.arange(C)).mean()), top1_text_to_centroid=float((S.argmax(0) == np.arange(C)).mean()),
                                  own_mean=float(np.diag(S).mean()), other_mean=float(S[~np.eye(C, dtype=bool)].mean()))
out['modality_gap'] = dict(norm_common_shift=float(np.linalg.norm(g)), mean_norm_of_gap_vectors=float(np.linalg.norm(gap, axis=1).mean()),
                           fraction_of_gap_energy_explained_by_common_shift=float(1 - (res ** 2).sum() / (gap ** 2).sum()),
                           cos_gap_with_common_shift_median=float(np.median(unit(gap) @ (g / np.linalg.norm(g)))),
                           residual_norm_mean=float(np.linalg.norm(res, axis=1).mean()), text_norm_mean=float(np.linalg.norm(T, axis=1).mean()), centroid_norm_mean=float(np.linalg.norm(mu, axis=1).mean()),
                           cos_text_minus_shift_vs_centroid_median=float(np.median(np.einsum('ij,ij->i', unit(T - g), unit(mu)))))
Xs = unit(X); out['accuracy'] = dict(nearest_true_centroid_cosine=float((np.argmax(Xs @ unit(mu).T, 1) == y).mean()), text_zero_shot_cosine=float((np.argmax(Xs @ unit(T).T, 1) == y).mean()))
json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1))
