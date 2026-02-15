#!/usr/bin/env python3
"""
Power-diagram (Laguerre) balanced K-means on GloVe embeddings.

Instead of adding an extra dimension, this uses a per-centroid scalar bias
that modifies the assignment rule:

    assign x_i to argmin_j  ||x_i - c_j||^2 - w_j

where w_j is a bias term updated each iteration to encourage balance:

    w_j -= lambda * (count_j - n/k)

Overpopulated centroids get lower w (larger effective distance),
underpopulated centroids get higher w (smaller effective distance).

This is equivalent to a power diagram / Laguerre tessellation and is
the geometric dual of optimal transport.
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


def compute_balance_stats(assignments, k):
    """Compute cluster balance statistics."""
    counts = np.bincount(assignments.astype(np.int64), minlength=k)
    mean_size = counts.mean()
    return {
        "mean": mean_size,
        "std": counts.std(),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "cv": counts.std() / mean_size,
        "max_over_mean": counts.max() / mean_size,
        "imbalance_factor": (counts.astype(np.float64) ** 2).sum() / (float(counts.sum()) ** 2) * k,
        "empty": int((counts == 0).sum()),
        "counts": counts,
    }


def standard_kmeans(vectors, k, niter=25, seed=42):
    """Run standard FAISS K-means as baseline."""
    n, d = vectors.shape
    km = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=seed)
    t0 = time.time()
    km.train(vectors)
    elapsed = time.time() - t0
    _, assignments = km.assign(vectors)
    obj = km.iteration_stats[-1]["obj"]
    return km.centroids, assignments, obj, elapsed


def power_kmeans(vectors, k, balance_lambda, niter=25, seed=42):
    """
    Power-diagram balanced K-means.

    Uses a per-centroid bias w_j updated each iteration:
        w_j -= balance_lambda * (count_j - n/k)

    Assignment: argmin_j ||x_i - c_j||^2 - w_j
    Centroid update: standard mean of assigned points.
    """
    n, d = vectors.shape
    rng = np.random.RandomState(seed)
    target_size = n / k

    # Initialize: random subset of data as centroids, zero biases
    perm = rng.permutation(n)[:k]
    centroids = vectors[perm].copy()
    w = np.zeros(k, dtype=np.float64)

    # Precompute ||x_i||^2 (constant, doesn't affect argmin but useful for obj)
    x_sq = (vectors ** 2).sum(axis=1)  # (n,)

    iter_stats = []
    t0 = time.time()

    for iteration in range(niter):
        # Compute all pairwise squared distances: (n, k)
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
        c_sq = (centroids ** 2).sum(axis=1)  # (k,)
        # Use BLAS for the x @ c^T part
        dots = vectors @ centroids.T  # (n, k)
        sq_dists = x_sq[:, None] - 2 * dots + c_sq[None, :]  # (n, k)

        # Power diagram assignment: subtract bias from distances
        biased_dists = sq_dists - w[None, :]  # (n, k)
        assignments = biased_dists.argmin(axis=1)

        # Compute true SSE (without bias, for comparison)
        true_dists = sq_dists[np.arange(n), assignments]
        sse = true_dists.sum()

        # Cluster sizes
        counts = np.bincount(assignments, minlength=k)

        iter_stats.append({
            "sse": float(sse),
            "cv": float(counts.std() / counts.mean()) if counts.mean() > 0 else float("inf"),
            "imbalance": float((counts.astype(np.float64) ** 2).sum() / (float(n) ** 2) * k),
            "empty": int((counts == 0).sum()),
            "max_size": int(counts.max()),
            "min_size": int(counts.min()),
        })

        # Update centroids: mean of assigned points
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = vectors[mask].mean(axis=0)
            else:
                # Reinitialize empty cluster from random point in largest cluster
                largest = counts.argmax()
                largest_mask = np.where(assignments == largest)[0]
                new_centroids[j] = vectors[largest_mask[rng.randint(len(largest_mask))]]
        centroids = new_centroids

        # Update biases: penalize overpopulation, reward underpopulation
        w -= balance_lambda * (counts - target_size)

    elapsed = time.time() - t0

    # Final assignment (unbiased, for fair SSE comparison)
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    unbiased_dists, unbiased_assignments = index.search(vectors, 1)
    unbiased_sse = unbiased_dists.sum()

    return {
        "centroids": centroids,
        "assignments": assignments,          # biased assignments (what the algorithm produces)
        "unbiased_assignments": unbiased_assignments.ravel(),  # nearest-centroid assignments
        "sse_biased": iter_stats[-1]["sse"],  # SSE of biased assignments
        "sse_unbiased": float(unbiased_sse),  # SSE of unbiased NN assignments
        "time": elapsed,
        "iter_stats": iter_stats,
        "biases": w.copy(),
    }


def run_experiments(vectors, k=256, niter=25):
    n, d = vectors.shape
    target_size = n / k

    print(f"\nDataset: {n} vectors, {d} dims, k={k}, target size={target_size:.1f}")

    # Baseline
    print("\n" + "=" * 70)
    print("STANDARD K-MEANS (baseline)")
    print("=" * 70)
    centroids_std, assign_std, obj_std, time_std = standard_kmeans(vectors, k, niter)
    stats_std = compute_balance_stats(assign_std, k)
    print(f"  SSE:          {obj_std:.2f}")
    print(f"  Time:         {time_std:.2f}s")
    print(f"  CV:           {stats_std['cv']:.4f}")
    print(f"  Imbalance:    {stats_std['imbalance_factor']:.4f}")
    print(f"  Min/Max size: {stats_std['min']} / {stats_std['max']}")

    # Power diagram with varying lambda
    lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []

    for lam in lambda_values:
        print(f"\n{'=' * 70}")
        print(f"POWER DIAGRAM K-MEANS (lambda={lam})")
        print("=" * 70)

        res = power_kmeans(vectors, k, balance_lambda=lam, niter=niter)
        stats_biased = compute_balance_stats(res["assignments"], k)
        stats_unbiased = compute_balance_stats(res["unbiased_assignments"], k)

        sse_increase = (res["sse_biased"] - obj_std) / obj_std * 100
        cv_reduction = (stats_std["cv"] - stats_biased["cv"]) / stats_std["cv"] * 100

        result = {
            "lambda": lam,
            "sse_biased": res["sse_biased"],
            "sse_unbiased": res["sse_unbiased"],
            "sse_increase_pct": sse_increase,
            "cv_biased": stats_biased["cv"],
            "cv_unbiased": stats_unbiased["cv"],
            "cv_reduction_pct": cv_reduction,
            "imbalance_biased": stats_biased["imbalance_factor"],
            "imbalance_unbiased": stats_unbiased["imbalance_factor"],
            "min_biased": stats_biased["min"],
            "max_biased": stats_biased["max"],
            "max_over_mean_biased": stats_biased["max_over_mean"],
            "empty_biased": stats_biased["empty"],
            "time": res["time"],
            "convergence": [(s["sse"], s["cv"]) for s in res["iter_stats"]],
        }
        results.append(result)

        print(f"  SSE (biased assign):   {res['sse_biased']:.2f} ({sse_increase:+.2f}% vs std)")
        print(f"  SSE (unbiased assign): {res['sse_unbiased']:.2f}")
        print(f"  Time:                  {res['time']:.2f}s")
        print(f"  --- Biased assignments (what the algorithm uses) ---")
        print(f"  CV:           {stats_biased['cv']:.4f} ({cv_reduction:+.1f}% reduction)")
        print(f"  Imbalance:    {stats_biased['imbalance_factor']:.4f}")
        print(f"  Min/Max size: {stats_biased['min']} / {stats_biased['max']}")
        print(f"  Empty:        {stats_biased['empty']}")
        print(f"  --- Convergence (first 5 iters) ---")
        for i, s in enumerate(res["iter_stats"][:5]):
            print(f"    iter {i:2d}: SSE={s['sse']:.0f}  CV={s['cv']:.4f}  "
                  f"sizes=[{s['min_size']}..{s['max_size']}]")
        if niter > 5:
            s = res["iter_stats"][-1]
            print(f"    iter {niter-1:2d}: SSE={s['sse']:.0f}  CV={s['cv']:.4f}  "
                  f"sizes=[{s['min_size']}..{s['max_size']}]")

    # Summary table
    print("\n\n" + "=" * 120)
    print("SUMMARY: Standard vs Power-Diagram K-means (k=256, GloVe-6B-50d)")
    print("=" * 120)
    hdr = (f"{'Method':>18} | {'SSE':>14} | {'SSE +%':>8} | {'CV':>8} | "
           f"{'CV Red%':>8} | {'Imbal':>7} | {'Min':>5} | {'Max':>5} | "
           f"{'Max/μ':>6} | {'Empty':>5} | {'Time':>7}")
    print(hdr)
    print("-" * 120)
    print(f"{'Standard':>18} | {obj_std:>14.2f} | {'---':>8} | {stats_std['cv']:>8.4f} | "
          f"{'---':>8} | {stats_std['imbalance_factor']:>7.4f} | {stats_std['min']:>5} | "
          f"{stats_std['max']:>5} | {stats_std['max_over_mean']:>6.2f} | "
          f"{stats_std['empty']:>5} | {time_std:>6.2f}s")
    for r in results:
        print(f"{'λ=' + str(r['lambda']):>18} | {r['sse_biased']:>14.2f} | "
              f"{r['sse_increase_pct']:>+7.2f}% | {r['cv_biased']:>8.4f} | "
              f"{r['cv_reduction_pct']:>+7.1f}% | {r['imbalance_biased']:>7.4f} | "
              f"{r['min_biased']:>5} | {r['max_biased']:>5} | "
              f"{r['max_over_mean_biased']:>6.2f} | {r['empty_biased']:>5} | "
              f"{r['time']:>6.2f}s")
    print("-" * 120)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "power_diagram_results.txt")
    with open(results_file, "w") as f:
        f.write("Power-Diagram Balanced K-means on GloVe-6B-50d\n")
        f.write(f"Dataset: {n} vectors, {d} dims, k={k}, niter={niter}\n")
        f.write("=" * 120 + "\n")
        f.write(hdr + "\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Standard':>18} | {obj_std:>14.2f} | {'---':>8} | {stats_std['cv']:>8.4f} | "
                f"{'---':>8} | {stats_std['imbalance_factor']:>7.4f} | {stats_std['min']:>5} | "
                f"{stats_std['max']:>5} | {stats_std['max_over_mean']:>6.2f} | "
                f"{stats_std['empty']:>5} | {time_std:>6.2f}s\n")
        for r in results:
            f.write(f"{'λ=' + str(r['lambda']):>18} | {r['sse_biased']:>14.2f} | "
                    f"{r['sse_increase_pct']:>+7.2f}% | {r['cv_biased']:>8.4f} | "
                    f"{r['cv_reduction_pct']:>+7.1f}% | {r['imbalance_biased']:>7.4f} | "
                    f"{r['min_biased']:>5} | {r['max_biased']:>5} | "
                    f"{r['max_over_mean_biased']:>6.2f} | {r['empty_biased']:>5} | "
                    f"{r['time']:>6.2f}s\n")
        f.write("-" * 120 + "\n")

        # Also save per-iteration convergence for the best lambda
        f.write("\n\nPER-ITERATION CONVERGENCE (all lambdas)\n")
        f.write("=" * 80 + "\n")
        for r in results:
            f.write(f"\nlambda={r['lambda']}:\n")
            f.write(f"  {'Iter':>4} | {'SSE':>14} | {'CV':>8} \n")
            f.write(f"  {'-' * 35}\n")
            for i, (sse, cv) in enumerate(r["convergence"]):
                f.write(f"  {i:>4} | {sse:>14.2f} | {cv:>8.4f}\n")

    print(f"\nResults saved to {results_file}")
    return results


def main():
    print("Power-Diagram Balanced K-means Experiment on GloVe Embeddings")
    print("=" * 70)
    vectors = load_glove()
    run_experiments(vectors, k=256, niter=25)
    print("\nDone!")


if __name__ == "__main__":
    main()
