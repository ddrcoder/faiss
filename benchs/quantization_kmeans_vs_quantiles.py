#!/usr/bin/env python3
"""
Existence proof: 1D k-means vs quantile-based quantization.

Two strategies for quantizing 1D data into k levels:
  1. K-means: optimize MSE directly (Lloyd's algorithm)
  2. Quantiles: partition into k equal-frequency bins, reconstruct with bin means

Evaluates both per-coordinate MSE and inner product reconstruction error
on GloVe-6B-50d (using a 50k subset for speed).
"""

import os
import time
import numpy as np

os.chdir("/tmp")
import faiss

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GLOVE_NPY = os.path.join(DATA_DIR, "glove.6B.50d.npy")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SUBSET_N = 50000
NUM_IP_PAIRS = 1000000  # pairs for inner product evaluation


def load_glove_subset():
    vectors = np.load(GLOVE_NPY)
    np.random.seed(42)
    idx = np.random.choice(vectors.shape[0], SUBSET_N, replace=False)
    vectors = vectors[idx]
    print(f"Using {vectors.shape[0]} vectors of dim {vectors.shape[1]}")
    return vectors


def quantile_reconstruct(x, k):
    """Quantile quantization: returns reconstructed values for each point."""
    n = len(x)
    idx = np.argsort(x)
    x_sorted = x[idx]
    bin_edges = np.linspace(0, n, k + 1, dtype=int)

    recon = np.empty(n, dtype=np.float32)
    for j in range(k):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        mean_val = x_sorted[lo:hi].astype(np.float64).mean()
        recon[idx[lo:hi]] = mean_val
    return recon


def kmeans_reconstruct(x, k, niter=20):
    """K-means quantization: returns reconstructed values for each point."""
    x_col = x.reshape(-1, 1).copy()
    kmeans = faiss.Kmeans(1, k, niter=niter, verbose=False)
    kmeans.train(x_col)
    D, I = kmeans.index.search(x_col, 1)
    centroids = kmeans.centroids.ravel()
    recon = centroids[I.ravel()].astype(np.float32)
    return recon


def evaluate_ip_error(true_vecs, recon_vecs, num_pairs):
    """Evaluate inner product reconstruction error on random pairs."""
    n = true_vecs.shape[0]
    np.random.seed(123)
    ai = np.random.randint(0, n, size=num_pairs)
    bi = np.random.randint(0, n, size=num_pairs)

    # True inner products
    true_ip = np.sum(true_vecs[ai].astype(np.float64) * true_vecs[bi].astype(np.float64), axis=1)
    # Reconstructed inner products
    recon_ip = np.sum(recon_vecs[ai].astype(np.float64) * recon_vecs[bi].astype(np.float64), axis=1)

    ip_errors = recon_ip - true_ip
    ip_mse = np.mean(ip_errors ** 2)
    ip_mae = np.mean(np.abs(ip_errors))
    ip_bias = np.mean(ip_errors)
    # Relative error: |recon - true| / |true|, excluding near-zero IPs
    mask = np.abs(true_ip) > 1e-3
    ip_rel = np.mean(np.abs(ip_errors[mask]) / np.abs(true_ip[mask]))

    return {
        "ip_mse": ip_mse,
        "ip_mae": ip_mae,
        "ip_bias": ip_bias,
        "ip_rel_error": ip_rel,
    }


def run_experiment(vectors):
    n, d = vectors.shape
    k_values = [4, 8, 16, 32, 64, 128, 256]

    print(f"Data: {n} points x {d} dimensions")
    print(f"Quantization levels: {k_values}")
    print(f"IP error evaluated on {NUM_IP_PAIRS:,} random pairs\n")

    # Header
    print(f"{'k':>5s}  {'---Coordinate MSE---':^25s}  {'-----Inner Product MSE-----':^30s}  "
          f"{'---IP Relative Error---':^25s}")
    print(f"{'':>5s}  {'Quantile':>10s} {'KMeans':>10s} {'KM/Q':>6s}  "
          f"{'Quantile':>12s} {'KMeans':>12s} {'KM/Q':>6s}  "
          f"{'Quantile':>10s} {'KMeans':>10s} {'KM/Q':>6s}")
    print("-" * 120)

    results = []
    for k in k_values:
        t0 = time.time()

        # Reconstruct all vectors under both methods
        q_recon = np.empty_like(vectors)
        km_recon = np.empty_like(vectors)

        for dim in range(d):
            col = vectors[:, dim]
            q_recon[:, dim] = quantile_reconstruct(col, k)
            km_recon[:, dim] = kmeans_reconstruct(col, k)

        # Per-coordinate MSE (averaged over all dims)
        q_coord_mse = np.mean((vectors.astype(np.float64) - q_recon.astype(np.float64)) ** 2)
        km_coord_mse = np.mean((vectors.astype(np.float64) - km_recon.astype(np.float64)) ** 2)

        # Inner product error
        q_ip = evaluate_ip_error(vectors, q_recon, NUM_IP_PAIRS)
        km_ip = evaluate_ip_error(vectors, km_recon, NUM_IP_PAIRS)

        elapsed = time.time() - t0

        coord_ratio = km_coord_mse / q_coord_mse
        ip_mse_ratio = km_ip["ip_mse"] / q_ip["ip_mse"]
        ip_rel_ratio = km_ip["ip_rel_error"] / q_ip["ip_rel_error"]

        print(f"{k:5d}  {q_coord_mse:10.6f} {km_coord_mse:10.6f} {coord_ratio:6.4f}  "
              f"{q_ip['ip_mse']:12.4f} {km_ip['ip_mse']:12.4f} {ip_mse_ratio:6.4f}  "
              f"{q_ip['ip_rel_error']:10.4f} {km_ip['ip_rel_error']:10.4f} {ip_rel_ratio:6.4f}  "
              f"({elapsed:.1f}s)")

        results.append({
            "k": k,
            "q_coord_mse": q_coord_mse,
            "km_coord_mse": km_coord_mse,
            "coord_ratio": coord_ratio,
            "q_ip_mse": q_ip["ip_mse"],
            "km_ip_mse": km_ip["ip_mse"],
            "ip_mse_ratio": ip_mse_ratio,
            "q_ip_rel": q_ip["ip_rel_error"],
            "km_ip_rel": km_ip["ip_rel_error"],
            "ip_rel_ratio": ip_rel_ratio,
            "q_ip_bias": q_ip["ip_bias"],
            "km_ip_bias": km_ip["ip_bias"],
            "q_ip_mae": q_ip["ip_mae"],
            "km_ip_mae": km_ip["ip_mae"],
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Ratio < 1 means k-means wins; ratio > 1 means quantiles win.\n")

    print("Additional detail — IP bias and MAE:")
    print(f"{'k':>5s}  {'---IP Bias---':^25s}  {'---IP MAE---':^25s}")
    print(f"{'':>5s}  {'Quantile':>10s} {'KMeans':>10s}  {'Quantile':>10s} {'KMeans':>10s}")
    for r in results:
        print(f"{r['k']:5d}  {r['q_ip_bias']:10.4f} {r['km_ip_bias']:10.4f}  "
              f"{r['q_ip_mae']:10.4f} {r['km_ip_mae']:10.4f}")

    return results


if __name__ == "__main__":
    vectors = load_glove_subset()
    results = run_experiment(vectors)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "kmeans_vs_quantiles.txt")
    with open(out_path, "w") as f:
        f.write("1D K-means vs Quantile Quantization — GloVe-6B-50d (50k subset)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Coordinate MSE and Inner Product reconstruction error\n\n")
        f.write(f"{'k':>5s}  {'Q CoordMSE':>11s} {'KM CoordMSE':>11s} {'Ratio':>6s}  "
                f"{'Q IP_MSE':>12s} {'KM IP_MSE':>12s} {'Ratio':>6s}  "
                f"{'Q IP_Rel':>9s} {'KM IP_Rel':>9s} {'Ratio':>6s}\n")
        for r in results:
            f.write(f"{r['k']:5d}  {r['q_coord_mse']:11.6f} {r['km_coord_mse']:11.6f} "
                    f"{r['coord_ratio']:6.4f}  {r['q_ip_mse']:12.4f} {r['km_ip_mse']:12.4f} "
                    f"{r['ip_mse_ratio']:6.4f}  {r['q_ip_rel']:9.4f} {r['km_ip_rel']:9.4f} "
                    f"{r['ip_rel_ratio']:6.4f}\n")
    print(f"\nResults saved to {out_path}")
