# One-shot separability of CLIP embeddings — experiments for the paper "One-Shot Separation Theorems"

Everything here reproduces the CLIP paragraph of the Introduction and the subsection
["One-shot separability of CLIP embeddings" of the Examples section (two figures, one table).](./files/clip_standalone.pdf)

## 0. Environment
Python >= 3.10, numpy, scipy, scikit-learn, matplotlib, nltk (WordNet is downloaded on first use);
for the embeddings only: torch, torchvision, open_clip_torch, tqdm.  One CPU core suffices for
everything except the embeddings (minutes on a GPU, ~15 min on a CPU for CIFAR-100).

## 1. Embeddings (notebooks)
* `compute_embeddings.ipynb`  — CLIP ViT-B/32 (OpenCLIP, checkpoint `laion2b_s34b_b79k`) embeddings of the
  10,000 CIFAR-100 test images (`X`, 10000x512), their labels (`y`), the embeddings of the 100 class-name prompts
  "a photo of a {}." (`T`, 100x512) and the class names  ->  `embeddings_cifar100.npz`
* `compute_text_embeddings.ipynb` — the same text encoder applied to the 1,000 ImageNet class names and to all
  WordNet noun lemmas (113,293)  ->  `text_embeddings.npz`
  Images are 32x32 and are upscaled to 224x224 by the OpenCLIP preprocessing; the sanity-check cell reports the
  zero-shot accuracy (75.8% for this checkpoint).

## 2. Scripts (run in this order; all take `--npz` / `--prep` / `--out` arguments)
| script | input | output | used for |
|---|---|---|---|
| `clip_auc.py` | embeddings_cifar100.npz | clip_auc.json, clip_auc_per_class.csv, **clip_auc_table.tex** | Table (AUC of the halfspace classifier); Introduction |
| `images_error_vs_dim.py` | embeddings_cifar100.npz | images_error_vs_dim.json | Figure 1 data |
| `prepare_labels.py` | text_embeddings.npz | labels_prepared.npz | dedup, cleaned vocabularies V1/V2 (WordNet synsets, nltk), PCA whitening |
| `labels_error_vs_dim.py` | labels_prepared.npz | labels_error_vs_dim.json | Figure 2 left |
| `labels_capacity.py` | labels_prepared.npz | labels_capacity.json | Figure 2 right (run `--dims` in parts if needed; n>=150 use `--trials 100`) |
| `labels_fullpass.py` | labels_prepared.npz | labels_fullpass.json | full-dimensional floor, violating pairs, examples |
| `centroids_vs_text.py` | embeddings_cifar100.npz | centroids_vs_text.json | text embedding vs class centroid remark |
| `make_figures.py` | the json files | **fig_oneshot_clip_pca.pdf**, **fig_labels_capacity.pdf** | figures |
`common.py` holds the shared definitions: the universal map h_alpha, the Fisher map, the one-shot error estimator,
PCA whitening and sphere-centring.

## 3. Geometry and estimator (as stated in the paper)
* Images: centred, PCA-whitened; dependence on n = retain the n leading components (eigenvalue order).
  The figure is cut where the condition number lambda_1/lambda_n reaches 100 (n=250; it reaches 10 at n=35).
* Labels: same for the dependence on n; the full-dimensional counts use sphere-projected, centred embeddings (no whitening).
* Estimator of c(P_n,1): fraction of ordered pairs (X_i,X_j), j!=i, with (X_j - X_i, c(X_i)) >= 0 over random anchors
  i (2,500 for images, 1,000 for labels) and all other points j.  Resolution = 1/(anchors*(N-1)).
* Capacity: largest m such that m random labels are all pairwise one-shot separable with probability >= 0.9
  (bisection over m, 200 Monte-Carlo trials; 100 for n>=150).

## 4. Headline numbers (this run)
AUC of (g(z),f(u)) per class: mean 0.982, min 0.914 (flatfish); per-class thresholds: 95.7% own / 6.4% other captured;
common threshold: balanced accuracy 0.927.  Image error decays as 0.839^n for n<=60; full space: 2 violating ordered
pairs = one duplicated image.  WordNet floor (sphere-centred, 512-d): 1.98% of labels (3,258 pairs); 1.11% one lemma per
synset; 0.41% without plant/animal taxa.  Capacity (cleaned vocabulary, whitened leading components): 4 (n=10), 54 (30),
425 (60), 1,098 (100), 2,580 (250).

## 5. LaTeX
`clip_intro_paragraph.tex`, `clip_examples_subsection.tex` (inputs `clip_auc_table.tex`), `clip_bib_entries.bib`.
