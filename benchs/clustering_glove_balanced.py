#!/usr/bin/env python3
"""
Balanced clustering experiment on GloVe embeddings.

Implements the "extra dimension penalty" approach to balanced K-means:
  - Append an extra dimension to all data vectors, set to 0
  - After each centroid update, set each centroid's extra dimension to
    a penalty proportional to its cluster population
  - This increases L2 distance to overpopulated centroids, encouraging
    more balanced assignments

Compares against standard K-means on GloVe-6B-50d.
"""

import os
import sys
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
        "cv": counts.std() / mean_size,  # coefficient of variation
        "max_over_mean": counts.max() / mean_size,
        "imbalance_factor": (counts.astype(np.float64) ** 2).sum() / (float(counts.sum()) ** 2) * k,
        "empty": int((counts == 0).sum()),
        "counts": counts,
    }


def standard_kmeans(vectors, k, niter=25, seed=42):
    """Run standard FAISS K-means."""
    n, d = vectors.shape
    km = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=seed)
    t0 = time.time()
    km.train(vectors)
    elapsed = time.time() - t0
    _, assignments = km.assign(vectors)
    obj = km.iteration_stats[-1]["obj"]
    return km.centroids, assignments, obj, elapsed


def balanced_kmeans_extra_dim(vectors, k, penalty_weight, niter=25, seed=42):
    """
    K-means with an extra penalty dimension for balance.

    The data gets an extra dimension set to 0. After each centroid update,
    each centroid's extra dimension is set to:
        penalty_weight * (cluster_size / mean_size - 1)

    So centroids of average-sized clusters get penalty=0, overpopulated
    clusters get positive penalty (increasing their distance), and
    underpopulated clusters get negative penalty (decreasing distance).
    """
    n, d = vectors.shape
    rng = np.random.RandomState(seed)

    # Append extra dimension (zeros) to data
    data_aug = np.zeros((n, d + 1), dtype=np.float32)
    data_aug[:, :d] = vectors

    # Initialize centroids: random subset + zero penalty dim
    perm = rng.permutation(n)[:k]
    centroids = np.zeros((k, d + 1), dtype=np.float32)
    centroids[:, :d] = vectors[perm]

    index = faiss.IndexFlatL2(d + 1)
    objectives = []
    t0 = time.time()

    for iteration in range(niter):
        # Assignment step: find nearest centroid for each point
        index.reset()
        index.add(centroids)
        distances, assignments = index.search(data_aug, 1)
        assignments = assignments.ravel()
        obj = distances.sum()
        objectives.append(obj)

        # Centroid update step
        new_centroids = np.zeros((k, d + 1), dtype=np.float32)
        counts = np.bincount(assignments.astype(np.int64), minlength=k)
        mean_size = max(counts[counts > 0].mean(), 1.0)

        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                # Update data dimensions from cluster members
                new_centroids[j, :d] = vectors[mask].mean(axis=0)
            else:
                # Reinitialize empty clusters from a random point
                new_centroids[j, :d] = vectors[rng.randint(n)]

        # Set penalty dimension based on cluster population
        for j in range(k):
            size_ratio = counts[j] / mean_size
            new_centroids[j, d] = penalty_weight * (size_ratio - 1.0)

        centroids = new_centroids

    elapsed = time.time() - t0

    # Final assignment (compute objective on original d-dimensional distances)
    orig_index = faiss.IndexFlatL2(d)
    orig_index.add(centroids[:, :d].copy())
    orig_distances, orig_assignments = orig_index.search(vectors, 1)
    true_obj = orig_distances.sum()

    return centroids[:, :d].copy(), orig_assignments.ravel(), true_obj, elapsed, objectives


def run_experiments(vectors, k=256, niter=25):
    """Run standard vs balanced K-means experiments."""
    n, d = vectors.shape
    mean_size = n / k

    print(f"\nDataset: {n} vectors, {d} dimensions, k={k}")
    print(f"Ideal cluster size: {mean_size:.1f}")

    # Standard K-means
    print("\n" + "=" * 70)
    print("STANDARD K-MEANS")
    print("=" * 70)
    centroids_std, assign_std, obj_std, time_std = standard_kmeans(vectors, k, niter)
    stats_std = compute_balance_stats(assign_std, k)
    print(f"  Objective (SSE):       {obj_std:.2f}")
    print(f"  Time:                  {time_std:.2f}s")
    print(f"  Cluster size mean:     {stats_std['mean']:.1f}")
    print(f"  Cluster size std:      {stats_std['std']:.1f}")
    print(f"  Cluster size min/max:  {stats_std['min']} / {stats_std['max']}")
    print(f"  CV (std/mean):         {stats_std['cv']:.4f}")
    print(f"  Max/mean ratio:        {stats_std['max_over_mean']:.2f}")
    print(f"  Imbalance factor:      {stats_std['imbalance_factor']:.4f}")
    print(f"  Empty clusters:        {stats_std['empty']}")

    # Balanced K-means with varying penalty weights
    penalty_weights = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = []

    for pw in penalty_weights:
        print(f"\n{'=' * 70}")
        print(f"BALANCED K-MEANS (penalty_weight={pw})")
        print("=" * 70)
        centroids_bal, assign_bal, obj_bal, time_bal, convergence = \
            balanced_kmeans_extra_dim(vectors, k, pw, niter)
        stats_bal = compute_balance_stats(assign_bal, k)

        obj_increase = (obj_bal - obj_std) / obj_std * 100
        cv_reduction = (stats_std["cv"] - stats_bal["cv"]) / stats_std["cv"] * 100

        result = {
            "penalty_weight": pw,
            "obj": obj_bal,
            "obj_increase_pct": obj_increase,
            "time": time_bal,
            "cv": stats_bal["cv"],
            "cv_reduction_pct": cv_reduction,
            "imbalance_factor": stats_bal["imbalance_factor"],
            "min_size": stats_bal["min"],
            "max_size": stats_bal["max"],
            "max_over_mean": stats_bal["max_over_mean"],
            "empty": stats_bal["empty"],
        }
        results.append(result)

        print(f"  Objective (SSE):       {obj_bal:.2f} ({obj_increase:+.2f}% vs standard)")
        print(f"  Time:                  {time_bal:.2f}s")
        print(f"  Cluster size mean:     {stats_bal['mean']:.1f}")
        print(f"  Cluster size std:      {stats_bal['std']:.1f}")
        print(f"  Cluster size min/max:  {stats_bal['min']} / {stats_bal['max']}")
        print(f"  CV (std/mean):         {stats_bal['cv']:.4f} ({cv_reduction:+.1f}% reduction)")
        print(f"  Max/mean ratio:        {stats_bal['max_over_mean']:.2f}")
        print(f"  Imbalance factor:      {stats_bal['imbalance_factor']:.4f}")
        print(f"  Empty clusters:        {stats_bal['empty']}")

    # Summary table
    print("\n\n" + "=" * 110)
    print("SUMMARY: Standard vs Balanced K-means (k=256, GloVe-6B-50d)")
    print("=" * 110)
    print(f"{'Method':>20} | {'SSE':>14} | {'SSE +%':>8} | {'CV':>8} | {'CV Red%':>8} | "
          f"{'Imbal':>7} | {'Min':>5} | {'Max':>5} | {'Max/μ':>6} | {'Time':>7}")
    print("-" * 110)
    print(f"{'Standard':>20} | {obj_std:>14.2f} | {'---':>8} | {stats_std['cv']:>8.4f} | "
          f"{'---':>8} | {stats_std['imbalance_factor']:>7.4f} | {stats_std['min']:>5} | "
          f"{stats_std['max']:>5} | {stats_std['max_over_mean']:>6.2f} | {time_std:>6.2f}s")
    for r in results:
        print(f"{'pw=' + str(r['penalty_weight']):>20} | {r['obj']:>14.2f} | {r['obj_increase_pct']:>+7.2f}% | "
              f"{r['cv']:>8.4f} | {r['cv_reduction_pct']:>+7.1f}% | {r['imbalance_factor']:>7.4f} | "
              f"{r['min_size']:>5} | {r['max_size']:>5} | {r['max_over_mean']:>6.2f} | {r['time']:>6.2f}s")
    print("-" * 110)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "balanced_clustering_results.txt")
    with open(results_file, "w") as f:
        f.write("Balanced K-means Experiment: Extra Dimension Penalty\n")
        f.write(f"Dataset: GloVe-6B-50d, {n} vectors, {d} dims, k={k}\n")
        f.write("=" * 110 + "\n")
        f.write(f"{'Method':>20} | {'SSE':>14} | {'SSE +%':>8} | {'CV':>8} | {'CV Red%':>8} | "
                f"{'Imbal':>7} | {'Min':>5} | {'Max':>5} | {'Max/μ':>6} | {'Time':>7}\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Standard':>20} | {obj_std:>14.2f} | {'---':>8} | {stats_std['cv']:>8.4f} | "
                f"{'---':>8} | {stats_std['imbalance_factor']:>7.4f} | {stats_std['min']:>5} | "
                f"{stats_std['max']:>5} | {stats_std['max_over_mean']:>6.2f} | {time_std:>6.2f}s\n")
        for r in results:
            f.write(f"{'pw=' + str(r['penalty_weight']):>20} | {r['obj']:>14.2f} | {r['obj_increase_pct']:>+7.2f}% | "
                    f"{r['cv']:>8.4f} | {r['cv_reduction_pct']:>+7.1f}% | {r['imbalance_factor']:>7.4f} | "
                    f"{r['min_size']:>5} | {r['max_size']:>5} | {r['max_over_mean']:>6.2f} | {r['time']:>6.2f}s\n")
        f.write("-" * 110 + "\n")
    print(f"\nResults saved to {results_file}")

    return results


def main():
    print("Balanced K-means Clustering Experiment on GloVe Embeddings")
    print("=" * 70)
    vectors = load_glove()
    run_experiments(vectors, k=256, niter=25)
    print("\nDone!")


if __name__ == "__main__":
    main()
