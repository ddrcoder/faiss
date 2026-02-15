#!/usr/bin/env python3
"""
Power-diagram balanced K-means v2 on GloVe embeddings.

Fixes from v1:
  - Proportional bias (set directly, not accumulated) to avoid drift
  - Warm-start from standard K-means centroids
  - Also test dual ascent with proper step sizing and bias centering

Assignment rule:  argmin_j  ||x_i - c_j||^2 - w_j

Three bias update strategies:
  A) Proportional:  w_j = alpha * (target - count_j)
  B) Dual ascent:   w_j += step * (target - count_j);  w -= mean(w)
  C) Log-proportional: w_j = alpha * log(target / max(count_j, 1))
"""

import os
import time
import numpy as np

os.chdir("/tmp")
import faiss

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GLOVE_NPY = os.path.join(DATA_DIR, "glove.6B.50d.npy")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_glove():
    vectors = np.load(GLOVE_NPY)
    print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    return vectors


def compute_stats(assignments, k):
    counts = np.bincount(assignments.astype(np.int64), minlength=k)
    mean_size = counts.mean()
    return {
        "mean": mean_size,
        "std": counts.std(),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "cv": counts.std() / mean_size if mean_size > 0 else float("inf"),
        "max_over_mean": counts.max() / mean_size if mean_size > 0 else float("inf"),
        "imbalance_factor": (counts.astype(np.float64) ** 2).sum() / (float(counts.sum()) ** 2) * k,
        "empty": int((counts == 0).sum()),
        "counts": counts,
    }


def standard_kmeans(vectors, k, niter=25, seed=42):
    n, d = vectors.shape
    km = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=seed)
    t0 = time.time()
    km.train(vectors)
    elapsed = time.time() - t0
    _, assignments = km.assign(vectors)
    obj = km.iteration_stats[-1]["obj"]
    return km.centroids.copy(), assignments, obj, elapsed


def biased_assign(vectors, centroids, w):
    """Assign each point to argmin_j ||x_i - c_j||^2 - w_j."""
    n, d = vectors.shape
    k = centroids.shape[0]
    x_sq = (vectors ** 2).sum(axis=1)
    c_sq = (centroids ** 2).sum(axis=1)
    dots = vectors @ centroids.T
    sq_dists = x_sq[:, None] - 2 * dots + c_sq[None, :]
    biased = sq_dists - w[None, :]
    assignments = biased.argmin(axis=1)
    true_dists = sq_dists[np.arange(n), assignments]
    return assignments, float(true_dists.sum())


def update_centroids(vectors, assignments, k, rng, old_centroids=None):
    """Compute new centroids as cluster means. Reinit empty clusters."""
    n, d = vectors.shape
    centroids = np.zeros((k, d), dtype=np.float32)
    counts = np.bincount(assignments.astype(np.int64), minlength=k)
    for j in range(k):
        mask = assignments == j
        if mask.sum() > 0:
            centroids[j] = vectors[mask].mean(axis=0)
        elif old_centroids is not None:
            # Split largest cluster
            largest = counts.argmax()
            largest_pts = np.where(assignments == largest)[0]
            centroids[j] = vectors[largest_pts[rng.randint(len(largest_pts))]]
        else:
            centroids[j] = vectors[rng.randint(n)]
    return centroids, counts


def power_kmeans_proportional(vectors, k, alpha, init_centroids, niter=25, seed=42):
    """
    Strategy A: Proportional bias.
    w_j = alpha * (target - count_j)
    Bias is set fresh each iteration — no accumulation, no drift.
    """
    n, d = vectors.shape
    rng = np.random.RandomState(seed)
    target = n / k
    centroids = init_centroids.copy()
    w = np.zeros(k, dtype=np.float64)

    iter_stats = []
    t0 = time.time()

    for it in range(niter):
        assignments, sse = biased_assign(vectors, centroids, w)
        counts = np.bincount(assignments.astype(np.int64), minlength=k)
        cv = counts.std() / counts.mean() if counts.mean() > 0 else float("inf")
        iter_stats.append({"sse": sse, "cv": cv, "min": int(counts.min()),
                           "max": int(counts.max()), "empty": int((counts == 0).sum())})

        centroids, counts = update_centroids(vectors, assignments, k, rng, centroids)

        # Proportional: set bias directly from current imbalance
        w = alpha * (target - counts.astype(np.float64))

    elapsed = time.time() - t0
    return {"assignments": assignments, "centroids": centroids, "sse": sse,
            "time": elapsed, "iter_stats": iter_stats, "biases": w}


def power_kmeans_dual(vectors, k, step, init_centroids, niter=25, seed=42):
    """
    Strategy B: Dual ascent with centering.
    w_j += step * (target - count_j)
    w -= mean(w)  # center to prevent drift
    """
    n, d = vectors.shape
    rng = np.random.RandomState(seed)
    target = n / k
    centroids = init_centroids.copy()
    w = np.zeros(k, dtype=np.float64)

    iter_stats = []
    t0 = time.time()

    for it in range(niter):
        assignments, sse = biased_assign(vectors, centroids, w)
        counts = np.bincount(assignments.astype(np.int64), minlength=k)
        cv = counts.std() / counts.mean() if counts.mean() > 0 else float("inf")
        iter_stats.append({"sse": sse, "cv": cv, "min": int(counts.min()),
                           "max": int(counts.max()), "empty": int((counts == 0).sum())})

        centroids, counts = update_centroids(vectors, assignments, k, rng, centroids)

        # Dual ascent with centering
        w += step * (target - counts.astype(np.float64))
        w -= w.mean()

    elapsed = time.time() - t0
    return {"assignments": assignments, "centroids": centroids, "sse": sse,
            "time": elapsed, "iter_stats": iter_stats, "biases": w}


def power_kmeans_log(vectors, k, alpha, init_centroids, niter=25, seed=42):
    """
    Strategy C: Log-proportional bias.
    w_j = alpha * log(target / max(count_j, 1))
    Softer correction for extreme imbalances.
    """
    n, d = vectors.shape
    rng = np.random.RandomState(seed)
    target = n / k
    centroids = init_centroids.copy()
    w = np.zeros(k, dtype=np.float64)

    iter_stats = []
    t0 = time.time()

    for it in range(niter):
        assignments, sse = biased_assign(vectors, centroids, w)
        counts = np.bincount(assignments.astype(np.int64), minlength=k)
        cv = counts.std() / counts.mean() if counts.mean() > 0 else float("inf")
        iter_stats.append({"sse": sse, "cv": cv, "min": int(counts.min()),
                           "max": int(counts.max()), "empty": int((counts == 0).sum())})

        centroids, counts = update_centroids(vectors, assignments, k, rng, centroids)

        # Log-proportional: softer correction
        safe_counts = np.maximum(counts, 1).astype(np.float64)
        w = alpha * np.log(target / safe_counts)

    elapsed = time.time() - t0
    return {"assignments": assignments, "centroids": centroids, "sse": sse,
            "time": elapsed, "iter_stats": iter_stats, "biases": w}


def print_result(label, res, obj_std, cv_std, k):
    stats = compute_stats(res["assignments"], k)
    sse_inc = (res["sse"] - obj_std) / obj_std * 100
    cv_red = (cv_std - stats["cv"]) / cv_std * 100
    print(f"\n  {label}")
    print(f"  SSE: {res['sse']:.2f} ({sse_inc:+.1f}%)  CV: {stats['cv']:.4f} ({cv_red:+.1f}%)  "
          f"Imbal: {stats['imbalance_factor']:.4f}")
    print(f"  Sizes: [{stats['min']}..{stats['max']}]  Max/μ: {stats['max_over_mean']:.2f}  "
          f"Empty: {stats['empty']}  Time: {res['time']:.2f}s")
    # Show convergence
    for i in [0, 4, 9, 14, 24]:
        if i < len(res["iter_stats"]):
            s = res["iter_stats"][i]
            print(f"    iter {i:2d}: SSE={s['sse']:.0f}  CV={s['cv']:.4f}  "
                  f"[{s['min']}..{s['max']}]  empty={s['empty']}")
    return stats, sse_inc, cv_red


def main():
    print("Power-Diagram Balanced K-means v2 on GloVe Embeddings")
    print("=" * 70)
    vectors = load_glove()
    n, d = vectors.shape
    k = 256
    niter = 25
    target = n / k
    print(f"\nk={k}, niter={niter}, target_size={target:.1f}")

    # Baseline
    print("\n" + "=" * 70)
    print("STANDARD K-MEANS (baseline)")
    print("=" * 70)
    init_centroids, assign_std, obj_std, time_std = standard_kmeans(vectors, k, niter)
    stats_std = compute_stats(assign_std, k)
    print(f"  SSE: {obj_std:.2f}  CV: {stats_std['cv']:.4f}  Imbal: {stats_std['imbalance_factor']:.4f}")
    print(f"  Sizes: [{stats_std['min']}..{stats_std['max']}]  Max/μ: {stats_std['max_over_mean']:.2f}  Time: {time_std:.2f}s")

    all_results = [("Standard", obj_std, stats_std["cv"], stats_std)]

    # Strategy A: Proportional
    print("\n" + "=" * 70)
    print("STRATEGY A: PROPORTIONAL BIAS  w_j = α*(target - count_j)")
    print("=" * 70)
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        res = power_kmeans_proportional(vectors, k, alpha, init_centroids, niter)
        label = f"Proportional α={alpha}"
        stats, sse_inc, cv_red = print_result(label, res, obj_std, stats_std["cv"], k)
        all_results.append((label, res["sse"], stats["cv"], stats))

    # Strategy B: Dual ascent with centering
    print("\n" + "=" * 70)
    print("STRATEGY B: DUAL ASCENT  w_j += step*(target - count_j); w -= mean(w)")
    print("=" * 70)
    for step in [0.001, 0.005, 0.01, 0.05]:
        res = power_kmeans_dual(vectors, k, step, init_centroids, niter)
        label = f"Dual step={step}"
        stats, sse_inc, cv_red = print_result(label, res, obj_std, stats_std["cv"], k)
        all_results.append((label, res["sse"], stats["cv"], stats))

    # Strategy C: Log-proportional
    print("\n" + "=" * 70)
    print("STRATEGY C: LOG-PROPORTIONAL  w_j = α*log(target/count_j)")
    print("=" * 70)
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        res = power_kmeans_log(vectors, k, alpha, init_centroids, niter)
        label = f"Log α={alpha}"
        stats, sse_inc, cv_red = print_result(label, res, obj_std, stats_std["cv"], k)
        all_results.append((label, res["sse"], stats["cv"], stats))

    # Final summary sorted by CV
    print("\n\n" + "=" * 110)
    print("FINAL SUMMARY (sorted by CV, lower = more balanced)")
    print("=" * 110)
    print(f"{'Method':>25} | {'SSE':>14} | {'SSE +%':>8} | {'CV':>8} | "
          f"{'Imbal':>7} | {'Min':>5} | {'Max':>5} | {'Max/μ':>6} | {'Empty':>5}")
    print("-" * 110)

    sorted_results = sorted(all_results, key=lambda x: x[2])
    for name, sse, cv, stats in sorted_results:
        sse_inc = (sse - obj_std) / obj_std * 100 if name != "Standard" else 0
        sse_str = f"{sse_inc:+7.1f}%" if name != "Standard" else "    ---"
        print(f"{name:>25} | {sse:>14.2f} | {sse_str:>8} | {stats['cv']:>8.4f} | "
              f"{stats['imbalance_factor']:>7.4f} | {stats['min']:>5} | {stats['max']:>5} | "
              f"{stats['max_over_mean']:>6.2f} | {stats['empty']:>5}")
    print("-" * 110)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "power_diagram_v2_results.txt")
    with open(results_file, "w") as f:
        f.write("Power-Diagram Balanced K-means v2 on GloVe-6B-50d\n")
        f.write(f"k={k}, niter={niter}, n={n}, d={d}\n")
        f.write("Warm-started from standard K-means centroids\n")
        f.write("=" * 110 + "\n")
        f.write(f"{'Method':>25} | {'SSE':>14} | {'SSE +%':>8} | {'CV':>8} | "
                f"{'Imbal':>7} | {'Min':>5} | {'Max':>5} | {'Max/μ':>6} | {'Empty':>5}\n")
        f.write("-" * 110 + "\n")
        for name, sse, cv, stats in sorted_results:
            sse_inc = (sse - obj_std) / obj_std * 100 if name != "Standard" else 0
            sse_str = f"{sse_inc:+7.1f}%" if name != "Standard" else "    ---"
            f.write(f"{name:>25} | {sse:>14.2f} | {sse_str:>8} | {stats['cv']:>8.4f} | "
                    f"{stats['imbalance_factor']:>7.4f} | {stats['min']:>5} | {stats['max']:>5} | "
                    f"{stats['max_over_mean']:>6.2f} | {stats['empty']:>5}\n")
        f.write("-" * 110 + "\n")
    print(f"\nResults saved to {results_file}")
    print("Done!")


if __name__ == "__main__":
    main()
