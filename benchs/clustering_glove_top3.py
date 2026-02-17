#!/usr/bin/env python3
"""
Top-3 weighted K-means on GloVe embeddings.

Variant of K-means where:
  - Each point is assigned to its 3 nearest centroids (soft assignment)
  - Centroids are recomputed as the weighted mean of all contributing points,
    weighted by 1/dist^2 (inverse square L2 distance)
  - The final evaluation uses standard hard assignment (nearest centroid)

Compares against standard K-means on GloVe-6B-50d.
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


def topk_weighted_kmeans(vectors, k, top_k=3, niter=25, seed=42):
    """
    K-means with top-k soft assignment, weighted by inverse square distance.

    Assignment: each point finds its top_k nearest centroids.
    Update: centroid_j = sum(w_ij * x_i) / sum(w_ij)
            where w_ij = 1 / (dist_ij^2 + eps)
            and the sum is over all points i that have j in their top-k.
    """
    n, d = vectors.shape
    rng = np.random.RandomState(seed)
    eps = 1e-10  # avoid division by zero

    # Initialize centroids from random subset
    perm = rng.permutation(n)[:k]
    centroids = vectors[perm].copy()

    index = faiss.IndexFlatL2(d)
    iter_stats = []
    t0 = time.time()

    for it in range(niter):
        # Assignment: find top_k nearest centroids for each point
        index.reset()
        index.add(centroids)
        distances, neighbors = index.search(vectors, top_k)
        # distances: (n, top_k), neighbors: (n, top_k)

        # Compute SSE using hard (top-1) assignment for tracking
        hard_sse = float(distances[:, 0].sum())

        # Compute weights: 1 / (dist^2 + eps)
        weights = 1.0 / (distances.astype(np.float64) + eps)  # distances are already squared in L2

        # Update centroids using weighted contributions
        new_centroids = np.zeros((k, d), dtype=np.float64)
        weight_sums = np.zeros(k, dtype=np.float64)

        for t in range(top_k):
            # For each rank t, scatter-add weighted vectors to their assigned centroid
            for j in range(k):
                mask = neighbors[:, t] == j
                if mask.any():
                    w = weights[mask, t]
                    new_centroids[j] += (vectors[mask].astype(np.float64) * w[:, None]).sum(axis=0)
                    weight_sums[j] += w.sum()

        # Normalize and handle empty clusters
        empty_count = 0
        for j in range(k):
            if weight_sums[j] > 0:
                new_centroids[j] /= weight_sums[j]
            else:
                # Reinit from random point
                new_centroids[j] = vectors[rng.randint(n)]
                empty_count += 1

        centroids = new_centroids.astype(np.float32)

        # Hard assignment stats
        hard_assignments = neighbors[:, 0]
        counts = np.bincount(hard_assignments.astype(np.int64), minlength=k)
        cv = counts.std() / counts.mean() if counts.mean() > 0 else float("inf")
        iter_stats.append({
            "sse": hard_sse, "cv": cv,
            "min": int(counts.min()), "max": int(counts.max()),
            "empty": empty_count,
        })

    elapsed = time.time() - t0

    # Final hard assignment
    index.reset()
    index.add(centroids)
    final_dists, final_assignments = index.search(vectors, 1)
    final_sse = float(final_dists.sum())
    final_assignments = final_assignments.ravel()

    return centroids, final_assignments, final_sse, elapsed, iter_stats


def topk_weighted_kmeans_fast(vectors, k, top_k=3, niter=25, seed=42):
    """
    Same algorithm but vectorized centroid update using np.add.at.
    """
    n, d = vectors.shape
    rng = np.random.RandomState(seed)
    eps = 1e-10

    perm = rng.permutation(n)[:k]
    centroids = vectors[perm].copy()

    index = faiss.IndexFlatL2(d)
    iter_stats = []
    t0 = time.time()

    for it in range(niter):
        index.reset()
        index.add(centroids)
        distances, neighbors = index.search(vectors, top_k)

        hard_sse = float(distances[:, 0].sum())

        # Weights: 1 / (sq_dist + eps).  faiss L2 returns squared distances.
        weights = 1.0 / (distances.astype(np.float64) + eps)

        # Scatter-add weighted contributions
        new_centroids = np.zeros((k, d), dtype=np.float64)
        weight_sums = np.zeros(k, dtype=np.float64)

        for t in range(top_k):
            w = weights[:, t]  # (n,)
            idx = neighbors[:, t]  # (n,)
            weighted_vecs = vectors.astype(np.float64) * w[:, None]  # (n, d)
            np.add.at(new_centroids, idx, weighted_vecs)
            np.add.at(weight_sums, idx, w)

        empty_count = 0
        for j in range(k):
            if weight_sums[j] > 0:
                new_centroids[j] /= weight_sums[j]
            else:
                new_centroids[j] = vectors[rng.randint(n)]
                empty_count += 1

        centroids = new_centroids.astype(np.float32)

        hard_assignments = neighbors[:, 0]
        counts = np.bincount(hard_assignments.astype(np.int64), minlength=k)
        cv = counts.std() / counts.mean() if counts.mean() > 0 else float("inf")
        iter_stats.append({
            "sse": hard_sse, "cv": cv,
            "min": int(counts.min()), "max": int(counts.max()),
            "empty": empty_count,
        })

    elapsed = time.time() - t0

    index.reset()
    index.add(centroids)
    final_dists, final_assignments = index.search(vectors, 1)
    final_sse = float(final_dists.sum())
    final_assignments = final_assignments.ravel()

    return centroids, final_assignments, final_sse, elapsed, iter_stats


def print_convergence(iter_stats):
    for i in [0, 4, 9, 14, 24]:
        if i < len(iter_stats):
            s = iter_stats[i]
            print(f"    iter {i:2d}: SSE={s['sse']:.0f}  CV={s['cv']:.4f}  "
                  f"[{s['min']}..{s['max']}]  empty={s['empty']}")


def main():
    print("Top-3 Weighted K-means on GloVe Embeddings")
    print("=" * 70)
    vectors = load_glove()
    n, d = vectors.shape
    niter = 25

    k_values = [256, 1024]
    all_results = []

    for k in k_values:
        target = n / k
        print(f"\n{'#' * 70}")
        print(f"# k = {k}  (target size = {target:.1f})")
        print(f"{'#' * 70}")

        # Standard K-means baseline
        print(f"\n{'=' * 70}")
        print("STANDARD K-MEANS")
        print("=" * 70)
        centroids_std, assign_std, obj_std, time_std = standard_kmeans(vectors, k, niter)
        stats_std = compute_stats(assign_std, k)
        print(f"  SSE: {obj_std:.2f}  Time: {time_std:.2f}s")
        print(f"  CV: {stats_std['cv']:.4f}  Sizes: [{stats_std['min']}..{stats_std['max']}]  "
              f"Imbal: {stats_std['imbalance_factor']:.4f}")

        # Top-k weighted K-means for different k values
        for top_k in [2, 3, 5]:
            print(f"\n{'=' * 70}")
            print(f"TOP-{top_k} WEIGHTED K-MEANS (1/dist² weights)")
            print("=" * 70)
            centroids_tk, assign_tk, obj_tk, time_tk, istats = \
                topk_weighted_kmeans_fast(vectors, k, top_k=top_k, niter=niter)
            stats_tk = compute_stats(assign_tk, k)
            sse_change = (obj_tk - obj_std) / obj_std * 100
            cv_change = (stats_tk["cv"] - stats_std["cv"]) / stats_std["cv"] * 100
            print(f"  SSE: {obj_tk:.2f} ({sse_change:+.2f}% vs standard)")
            print(f"  Time: {time_tk:.2f}s")
            print(f"  CV: {stats_tk['cv']:.4f} ({cv_change:+.1f}%)  "
                  f"Sizes: [{stats_tk['min']}..{stats_tk['max']}]  "
                  f"Imbal: {stats_tk['imbalance_factor']:.4f}")
            print(f"  Convergence:")
            print_convergence(istats)

            all_results.append({
                "k": k, "method": f"Top-{top_k} 1/d²",
                "sse": obj_tk, "sse_change": sse_change,
                "cv": stats_tk["cv"], "cv_change": cv_change,
                "imbal": stats_tk["imbalance_factor"],
                "min": stats_tk["min"], "max": stats_tk["max"],
                "time": time_tk,
            })

        all_results.append({
            "k": k, "method": "Standard",
            "sse": obj_std, "sse_change": 0.0,
            "cv": stats_std["cv"], "cv_change": 0.0,
            "imbal": stats_std["imbalance_factor"],
            "min": stats_std["min"], "max": stats_std["max"],
            "time": time_std,
        })

    # Summary
    print("\n\n" + "=" * 120)
    print("SUMMARY: Standard vs Top-k Weighted K-means on GloVe-6B-50d")
    print("=" * 120)
    print(f"{'k':>6} | {'Method':>18} | {'SSE':>14} | {'SSE Δ%':>8} | {'CV':>8} | "
          f"{'CV Δ%':>8} | {'Imbal':>7} | {'Min':>5} | {'Max':>5} | {'Time':>7}")
    print("-" * 120)
    for r in sorted(all_results, key=lambda x: (x["k"], x["method"])):
        sse_str = f"{r['sse_change']:+7.2f}%" if r["method"] != "Standard" else "    ---"
        cv_str = f"{r['cv_change']:+7.1f}%" if r["method"] != "Standard" else "    ---"
        print(f"{r['k']:>6} | {r['method']:>18} | {r['sse']:>14.2f} | {sse_str:>8} | "
              f"{r['cv']:>8.4f} | {cv_str:>8} | {r['imbal']:>7.4f} | "
              f"{r['min']:>5} | {r['max']:>5} | {r['time']:>6.2f}s")
    print("-" * 120)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "top3_weighted_results.txt")
    with open(results_file, "w") as f:
        f.write("Top-k Weighted K-means on GloVe-6B-50d\n")
        f.write(f"n={n}, d={d}, niter={niter}\n")
        f.write("Weight = 1/(squared_L2_dist + eps)\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'k':>6} | {'Method':>18} | {'SSE':>14} | {'SSE Δ%':>8} | {'CV':>8} | "
                f"{'CV Δ%':>8} | {'Imbal':>7} | {'Min':>5} | {'Max':>5} | {'Time':>7}\n")
        f.write("-" * 120 + "\n")
        for r in sorted(all_results, key=lambda x: (x["k"], x["method"])):
            sse_str = f"{r['sse_change']:+7.2f}%" if r["method"] != "Standard" else "    ---"
            cv_str = f"{r['cv_change']:+7.1f}%" if r["method"] != "Standard" else "    ---"
            f.write(f"{r['k']:>6} | {r['method']:>18} | {r['sse']:>14.2f} | {sse_str:>8} | "
                    f"{r['cv']:>8.4f} | {cv_str:>8} | {r['imbal']:>7.4f} | "
                    f"{r['min']:>5} | {r['max']:>5} | {r['time']:>6.2f}s\n")
        f.write("-" * 120 + "\n")
    print(f"\nResults saved to {results_file}")
    print("Done!")


if __name__ == "__main__":
    main()
