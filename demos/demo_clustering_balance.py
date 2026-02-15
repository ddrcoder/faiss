"""
Experiment: K-means vs K-medians vs Hybrid clustering on a 2D Gaussian.

Compares four approaches:
  1. K-means:    L2 assignment, mean centroids
  2. K-medians:  L1 assignment, component-wise median centroids
  3. Hybrid:     L1 assignment (median centroids for iteration),
                 L2 reconstruction centroids (cluster means)
  4. Hybrid2:    L1 assignment, mean centroids for both iteration and recon

Measures:
  - Balance:    max/min cluster size ratio, std of cluster sizes
  - Distortion: MSE (L2 reconstruction error to cluster mean)
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time


def assign_l2(X, centroids):
    dists = cdist(X, centroids, metric="sqeuclidean")
    return np.argmin(dists, axis=1)


def assign_l1(X, centroids):
    dists = cdist(X, centroids, metric="cityblock")
    return np.argmin(dists, axis=1)


def centroids_mean(X, labels, k):
    d = X.shape[1]
    centroids = np.zeros((k, d), dtype=np.float64)
    counts = np.bincount(labels, minlength=k)
    for dim in range(d):
        sums = np.bincount(labels, weights=X[:, dim], minlength=k)
        mask = counts > 0
        centroids[mask, dim] = sums[mask] / counts[mask]
    return centroids.astype(X.dtype)


def centroids_median(X, labels, k):
    d = X.shape[1]
    centroids = np.zeros((k, d), dtype=X.dtype)
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            centroids[j] = np.median(X[mask], axis=0)
    return centroids


def compute_mse(X, labels, centroids):
    residuals = X - centroids[labels]
    return np.mean(np.sum(residuals ** 2, axis=1))


def balance_stats(labels, k):
    counts = np.bincount(labels, minlength=k)
    nonempty = counts[counts > 0]
    return {
        "max_min_ratio": nonempty.max() / nonempty.min(),
        "std": np.std(counts),
        "min": int(nonempty.min()),
        "max": int(nonempty.max()),
        "empty": int(np.sum(counts == 0)),
        "counts": counts,
    }


def kmeans_pp_init(X, k, seed=42):
    """K-means++ with incremental distance tracking."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
    idx = rng.randint(n)
    centroids[0] = X[idx]
    # Track min distance to any chosen centroid
    min_d2 = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = min_d2 / min_d2.sum()
        chosen = rng.choice(n, p=probs)
        centroids[i] = X[chosen]
        new_d2 = np.sum((X - centroids[i]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, new_d2)
    return centroids


def run_method(X, k, assign_fn, update_fn, n_iter=50, seed=42):
    centroids = kmeans_pp_init(X, k, seed)
    for _ in range(n_iter):
        labels = assign_fn(X, centroids)
        centroids = update_fn(X, labels, k)
    recon = centroids_mean(X, labels, k)
    mse = compute_mse(X, labels, recon)
    bal = balance_stats(labels, k)
    return labels, centroids, recon, mse, bal


def main():
    np.random.seed(42)
    n = 100_000
    k = 256

    print(f"2D standard Gaussian: n={n}, k={k}\n")
    X = np.random.randn(n, 2).astype(np.float32)

    methods = [
        ("K-means (L2/mean)",       assign_l2, centroids_mean),
        ("K-medians (L1/median)",   assign_l1, centroids_median),
        ("Hybrid (L1/median iter)", assign_l1, centroids_median),
        ("Hybrid2 (L1/mean iter)",  assign_l1, centroids_mean),
    ]
    # Note: "Hybrid" and "K-medians" use identical iteration;
    # they differ only in that we always report mean-based MSE.
    # The distinction is conceptual: Hybrid emphasizes that
    # reconstruction uses the cluster mean, not the median.

    results = {}
    for name, afn, ufn in methods:
        print(f"Running {name}...", end=" ", flush=True)
        t0 = time.time()
        labels, assign_c, recon_c, mse, bal = run_method(X, k, afn, ufn)
        elapsed = time.time() - t0
        results[name] = {
            "labels": labels, "assign_centroids": assign_c,
            "recon_centroids": recon_c, "mse": mse,
            "balance": bal, "time": elapsed,
        }
        print(f"{elapsed:.1f}s")
        print(f"  MSE: {mse:.6f}")
        print(f"  Sizes: min={bal['min']} max={bal['max']} "
              f"std={bal['std']:.1f} max/min={bal['max_min_ratio']:.2f} "
              f"empty={bal['empty']}\n")

    # Summary
    print("=" * 78)
    print(f"{'Method':<30} {'MSE':>8} {'Max/Min':>8} {'Std':>8} "
          f"{'Min':>5} {'Max':>5} {'Empty':>6}")
    print("-" * 78)
    for name, r in results.items():
        b = r["balance"]
        print(f"{name:<30} {r['mse']:>8.5f} {b['max_min_ratio']:>8.2f} "
              f"{b['std']:>8.1f} {b['min']:>5} {b['max']:>5} {b['empty']:>6}")
    print("=" * 78)
    print(f"Ideal cluster size: {n // k}")

    # --- Scatter plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx, (name, r) in enumerate(results.items()):
        ax = axes.ravel()[idx]
        lab = r["labels"]
        rc = r["recon_centroids"]
        ac = r["assign_centroids"]
        counts = r["balance"]["counts"]

        sub = np.random.choice(len(X), 5000, replace=False)
        ax.scatter(X[sub, 0], X[sub, 1], c=lab[sub], cmap="tab20",
                   s=1, alpha=0.3)
        # Reconstruction centroids in red
        sizes = counts / max(counts.max(), 1) * 80 + 5
        ax.scatter(rc[:, 0], rc[:, 1], c="red", s=sizes, marker="x",
                   linewidths=0.8, zorder=5)
        # Show offset between assign and recon centroids if different
        if not np.allclose(ac, rc):
            for j in range(k):
                if counts[j] > 0:
                    ax.plot([ac[j, 0], rc[j, 0]], [ac[j, 1], rc[j, 1]],
                            "k-", alpha=0.15, linewidth=0.5)

        b = r["balance"]
        ax.set_title(f"{name}\nMSE={r['mse']:.5f}  Max/Min={b['max_min_ratio']:.2f}"
                     f"  Std={b['std']:.1f}", fontsize=10)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Clustering Balance: 2D Gaussian, n={n}, k={k}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/home/user/faiss/demos/clustering_balance_comparison.png", dpi=150)
    print("\nSaved: demos/clustering_balance_comparison.png")

    # --- Histograms ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    for idx, (name, r) in enumerate(results.items()):
        ax = axes2.ravel()[idx]
        counts = r["balance"]["counts"]
        ax.hist(counts, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(n // k, color="red", linestyle="--", label=f"ideal={n // k}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Cluster size"); ax.set_ylabel("Count"); ax.legend()
        b = r["balance"]
        ax.text(0.95, 0.95, f"std={b['std']:.1f}\nmin={b['min']}\nmax={b['max']}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.suptitle("Distribution of Cluster Sizes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/home/user/faiss/demos/clustering_balance_histograms.png", dpi=150)
    print("Saved: demos/clustering_balance_histograms.png")


if __name__ == "__main__":
    main()
