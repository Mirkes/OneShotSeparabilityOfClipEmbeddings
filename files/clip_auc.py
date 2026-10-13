#!/usr/bin/env python3
"""
clip_auc.py -- the zero-shot rule (g(z), f(u)) >= theta as a halfspace classifier.

For every CIFAR-100 class z, the score (g(z), f(u)) on the RAW CLIP embeddings
(no normalisation, no centring) is used to separate the images of class z
from the images of all other classes.  Reported per class: ROC AUC, the
threshold theta_z maximising balanced accuracy, and the recall / false-inclusion
rate at that threshold.  Also reported: the best single threshold common to all
classes, and the accuracy of the argmax rule (raw inner products and cosine).

Input : embeddings_cifar100.npz with arrays X [N,512], y [N], T [100,512], classes [100]
Output: clip_auc.json, clip_auc_per_class.csv, clip_auc_table.tex
"""
import argparse, json, csv
import numpy as np
from sklearn.metrics import roc_auc_score


def best_threshold(score, positive):
    """Threshold maximising balanced accuracy; returns (theta, bal_acc, recall, fpr)."""
    order = np.argsort(-score)
    lab = positive[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    bal = 0.5 * (tp / lab.sum() + 1.0 - fp / (~lab).sum())
    i = int(bal.argmax())
    return float(score[order][i]), float(bal[i]), float(tp[i] / lab.sum()), float(fp[i] / (~lab).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="embeddings_cifar100.npz")
    ap.add_argument("--out", default="clip_auc")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X, y, T, classes = d["X"].astype(np.float64), d["y"], d["T"].astype(np.float64), np.array(d["classes"])
    C = len(classes)
    S = X @ T.T                                   # raw inner products (g(z), f(u)), no normalisation, no centring

    auc = np.array([roc_auc_score(y == z, S[:, z]) for z in range(C)])
    per = np.array([best_threshold(S[:, z], y == z) for z in range(C)])   # theta_z, bal, recall, fpr

    # one common threshold: maximise the mean balanced accuracy over classes
    grid = np.linspace(np.percentile(S, 50), np.percentile(S, 99.9), 400)
    bal_common = [np.mean([0.5 * ((S[y == z, z] > c).mean() + (S[y != z, z] <= c).mean()) for z in range(C)]) for c in grid]
    i = int(np.argmax(bal_common))
    theta_common, bal_common_best = float(grid[i]), float(bal_common[i])
    rec_common = float(np.mean([(S[y == z, z] > theta_common).mean() for z in range(C)]))
    fpr_common = float(np.mean([(S[y != z, z] > theta_common).mean() for z in range(C)]))

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True); Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    acc_raw = float(np.mean(S.argmax(1) == y)); acc_cos = float(np.mean((Xn @ Tn.T).argmax(1) == y))

    order = np.argsort(auc)
    summary = dict(
        n_images=int(len(X)), n_classes=int(C),
        auc=dict(mean=float(auc.mean()), median=float(np.median(auc)), q10=float(np.percentile(auc, 10)),
                 min=float(auc.min()), max=float(auc.max()), n_above_0_95=int((auc > 0.95).sum())),
        per_class_threshold=dict(balanced_accuracy=float(per[:, 1].mean()), recall=float(per[:, 2].mean()),
                                 false_inclusion=float(per[:, 3].mean()), theta_mean=float(per[:, 0].mean()),
                                 theta_cv=float(per[:, 0].std() / per[:, 0].mean())),
        common_threshold=dict(theta=theta_common, balanced_accuracy=bal_common_best, recall=rec_common, false_inclusion=fpr_common),
        argmax_accuracy=dict(raw_inner_product=acc_raw, cosine=acc_cos),
        lowest_auc=[(str(classes[k]), float(auc[k])) for k in order[:5]],
        highest_auc=[(str(classes[k]), float(auc[k])) for k in order[-5:][::-1]],
    )
    json.dump(summary, open(args.out + ".json", "w"), indent=1)

    with open(args.out + "_per_class.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "auc", "theta_z", "balanced_accuracy", "recall", "false_inclusion"])
        for z in range(C):
            w.writerow([classes[z], f"{auc[z]:.4f}", f"{per[z,0]:.3f}", f"{per[z,1]:.4f}", f"{per[z,2]:.4f}", f"{per[z,3]:.4f}"])

    # LaTeX table: distribution of AUC and the two threshold regimes
    q = np.percentile(auc, [0, 10, 25, 50, 75, 90, 100])
    tex = []
    tex.append(r"\begin{table}[tb]")
    tex.append(r"\centering")
    tex.append(r"\begin{tabular}{lccccccc}")
    tex.append(r"\hline")
    tex.append(r"ROC AUC over the 100 classes & min & 10\% & 25\% & median & 75\% & 90\% & max \\")
    tex.append(r"\hline")
    tex.append(r" & " + " & ".join(f"{v:.3f}" for v in q) + r" \\")
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\\[2mm]")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\hline")
    tex.append(r"Halfspace $\{x:(g(z),x)\ge\theta\}$ & balanced accuracy & own images captured & other images captured \\")
    tex.append(r"\hline")
    tex.append(rf"threshold $\theta_z$ chosen per class & {per[:,1].mean():.3f} & {100*per[:,2].mean():.1f}\% & {100*per[:,3].mean():.1f}\% \\")
    tex.append(rf"one threshold $\theta$ for all classes & {bal_common_best:.3f} & {100*rec_common:.1f}\% & {100*fpr_common:.1f}\% \\")
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    lo = ", ".join(f"``{n.replace('_',' ')}'' ({a:.3f})" for n, a in summary["lowest_auc"][:3])
    hi = ", ".join(f"``{n.replace('_',' ')}'' ({a:.3f})" for n, a in summary["highest_auc"][:3])
    tex.append(r"\caption{The zero-shot rule as a halfspace classifier on the raw CLIP embeddings of the CIFAR-100 test set ($10{,}000$ images, $100$ classes). "
               r"Top: distribution over classes of the ROC AUC of the score $(g(z),f(u))$ for separating the images of class $z$ from all others; "
               rf"the lowest values are {lo}, the highest {hi}. "
               r"Bottom: balanced accuracy, recall and false-inclusion rate of the halfspace with the threshold maximising balanced accuracy, chosen per class or common to all classes. "
               rf"The argmax rule over the $100$ classes has accuracy {100*acc_raw:.1f}\% with raw inner products and {100*acc_cos:.1f}\% with cosine similarity.}}")
    tex.append(r"\label{tab:clip-auc}")
    tex.append(r"\end{table}")
    open(args.out + "_table.tex", "w").write("\n".join(tex) + "\n")

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
