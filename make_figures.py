#!/usr/bin/env python3
"""Draw fig_oneshot_clip_pca.pdf (images) and fig_labels_capacity.pdf (labels) from the JSON results."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
nan = lambda v: [x if x > 0 else np.nan for x in v]

# ---------------- Figure 1: images
r = json.load(open('images_error_vs_dim.json')); rows = r['rows']; ns = np.array([x['n'] for x in rows]); eu = np.array([x['univ'] for x in rows])
a, b = r['fit_intercept'], np.log(r['fit_factor']); n10, n100 = r['n_cond10'], r['n_cond100']
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.semilogy(ns, nan(eu), 'o-', color='#1f4e79', label='CLIP, $n$ leading principal components, universal map')
ax.semilogy(ns, nan([x['fisher'] for x in rows]), 's--', color='#1f4e79', alpha=.55, label='CLIP, $n$ leading principal components, Fisher normal')
ax.semilogy(ns, nan([x['gauss'] for x in rows]), '^-', color='#2f7d32', label='i.i.d. Gaussian, same $N$')
ax.semilogy(ns, nan([x['laplace'] for x in rows]), 'v-', color='#b5651d', label='i.i.d. Laplace, same $N$')
nn = np.arange(5, 65)
ax.semilogy(nn, 2.0 ** (-nn / 2), ':', color='#2f7d32', label=r'bound (27): Gaussian $(1/\sqrt{2})^{n}$')
ax.semilogy(nn, 0.7872 ** nn, ':', color='#b5651d', label=r'bound (27): Laplace $0.7872^{\,n}$')
ax.semilogy(nn, np.exp(a + b * nn), '-.', color='gray', label=r'fit to CLIP, $n\leq 60$: $\propto %.3f^{\,n}$' % r['fit_factor'])
ax.axhline(r['resolution'], color='k', lw=.6, ls='--'); ax.text(150, 1.6 * r['resolution'], 'resolution of the estimator', fontsize=7)
for nc, lab in [(n10, r'$\lambda_1/\lambda_n=10$'), (n100, r'$\lambda_1/\lambda_n=100$')]:
    ax.axvline(nc, color='#7b1f3a', lw=.8, ls='--'); ax.text(nc - 2 if nc > 100 else nc + 2, 2e-9, f'{lab}\n($n={nc}$)', fontsize=7, color='#7b1f3a', va='bottom', ha='right' if nc > 100 else 'left')
ax.set_xlim(0, n100 + 5); ax.set_ylim(1e-9, 0.3); ax.set_xticks(range(0, n100 + 1, 50))
ax.set_xlabel('number of leading principal components $n$ (whitened)'); ax.set_ylabel(r'empirical one-shot error $\hat c(\mathbb{P}_n,1)$')
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(0.985, 0.985))
ax.set_title('One-shot separability of CLIP ViT-B/32 image embeddings (CIFAR-100 test, $N=10{,}000$)', fontsize=9)
fig.tight_layout(); fig.savefig('fig_oneshot_clip_pca.pdf'); fig.savefig('fig_oneshot_clip_pca.png', dpi=160)

# ---------------- Figure 2: labels
L = json.load(open('labels_error_vs_dim.json')); R = L['results']; C = json.load(open('labels_capacity.json'))['results']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
for key, lab, col, mk in [('wordnet_all', 'WordNet nouns, 112,058 distinct labels', '#7b1f3a', 'o'), ('wordnet_V2', 'WordNet, cleaned vocabulary (51,911 concepts)', '#c8782a', 's'), ('imagenet1k', 'ImageNet-1k class names', '#2f7d32', 'd')]:
    ax1.semilogy([x['n'] for x in R[key]], nan([x['err'] for x in R[key]]), mk + '-', color=col, label=lab)
ax1.semilogy(ns, nan(eu), '^-', color='#1f4e79', label='CIFAR-100 image embeddings (Figure 1)')
ax1.axhline(L['resolution_wordnet'], color='k', lw=.6, ls='--'); ax1.text(120, 1.5 * L['resolution_wordnet'], 'resolution (WordNet)', fontsize=7)
ax1.set_xlim(0, 255); ax1.set_ylim(1e-9, 0.1); ax1.set_xticks(range(0, 251, 50)); ax1.grid(alpha=.3, which='both'); ax1.legend(fontsize=7, loc='upper right')
ax1.set_xlabel('number of leading principal components $n$ (whitened)'); ax1.set_ylabel(r'empirical one-shot error $\hat c(\mathbb{P}_n,1)$, universal map')
ax1.set_title('Label embeddings: error vs dimension', fontsize=9)
nn = sorted(int(k) for k in C)
ax2.semilogy(nn, [C[str(n)]['univ']['capacity'] for n in nn], 'o-', color='#7b1f3a', label=r'empirical capacity $m^*$ ($\delta=0.1$), universal map')
ax2.semilogy(nn, [C[str(n)]['fisher']['capacity'] for n in nn], 's--', color='#7b1f3a', alpha=.5, label=r'empirical capacity $m^*$, Fisher normal')
pred = [C[str(n)]['univ']['pred'] for n in nn]; ax2.semilogy([n for n, p in zip(nn, pred) if np.isfinite(p)], [p for p in pred if np.isfinite(p)], 'x:', color='gray', label=r'Proposition 3.1: $\sqrt{\delta/\hat c}$')
for n in nn: ax2.annotate(str(C[str(n)]['univ']['capacity']), (n, C[str(n)]['univ']['capacity']), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7)
ax2.set_xlim(0, 255); ax2.set_xticks(range(0, 251, 50)); ax2.grid(alpha=.3, which='both'); ax2.legend(fontsize=7, loc='lower right')
ax2.set_xlabel('number of leading principal components $n$ (whitened)'); ax2.set_ylabel('capacity $m^*$ (labels)')
ax2.set_title(r'Largest $m$ such that $m$ random labels are all one-shot separable w.p. $\geq 0.9$', fontsize=9)
fig.tight_layout(); fig.savefig('fig_labels_capacity.pdf'); fig.savefig('fig_labels_capacity.png', dpi=160)
print('figures written')
