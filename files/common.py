"""Shared helpers for the CLIP one-shot separability experiments."""
import numpy as np

def h_alpha(x, alpha=1.0):
    """Universal map of Theorem 5.6: c_i = alpha x_i / (1 + alpha |x_i|)."""
    return alpha * x / (1.0 + alpha * np.abs(x))

def fisher(x, alpha=1.0):
    return alpha * x

def oneshot_error(Z, anchors, cmap=h_alpha, alpha=1.0, batch=500):
    """Fraction of ordered pairs (X_i, X_j), j != i, with (X_j - alpha X_i, c(alpha X_i)) >= 0,
    over the given anchors i and all other points j.  Returns the mean over anchors."""
    Z = np.asarray(Z, dtype=np.float32); total = 0.0
    for s in range(0, len(anchors), batch):
        idx = anchors[s:s + batch]; A = Z[idx]
        Cn = cmap(alpha * A).astype(np.float32); thr = alpha * np.einsum('bn,bn->b', A, Cn)
        V = (Z @ Cn.T) >= thr[None, :]
        V[idx, np.arange(len(idx))] = False
        total += V.sum()
    return total / (len(anchors) * (len(Z) - 1))

def pca_whiten(A):
    """Centre and whiten by principal components.  Returns scores P (columns ordered by eigenvalue,
    unit variance) and the eigenvalues of the covariance."""
    A = np.asarray(A, dtype=np.float64); A = A - A.mean(0)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    lam = S ** 2 / (len(A) - 1)
    return (U * np.sqrt(len(A) - 1)).astype(np.float32), lam

def sphere_centre(A):
    """Project onto the unit sphere, then subtract the mean."""
    A = np.asarray(A, dtype=np.float64); A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return (A - A.mean(0)).astype(np.float32)
