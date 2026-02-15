#!/usr/bin/env python3
"""
Clustering experiments on GloVe word embeddings using FAISS.

Experiment 1: K-means clustering with varying k on GloVe-6B-50d
  - Runs k-means for k in {16, 64, 256, 1024, 4096}
  - Reports: final objective, per-iteration convergence, time, empty clusters

Experiment 2: Effect of niter and nredo on clustering quality
  - Fixes k=256, varies niter in {5, 10, 25, 50} and nredo in {1, 3, 5}
  - Reports: final objective, total time

Downloads GloVe-6B (50d) automatically if not already present.
"""

import os
import sys
import time
import zipfile
import urllib.request
import numpy as np

# Run from /tmp so the local faiss source directory doesn't shadow the installed package
os.chdir("/tmp")
import faiss


GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GLOVE_ZIP = os.path.join(DATA_DIR, "glove.6B.zip")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.6B.50d.txt")
GLOVE_NPY = os.path.join(DATA_DIR, "glove.6B.50d.npy")
GLOVE_WORDS = os.path.join(DATA_DIR, "glove.6B.50d.words.npy")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def download_glove():
    """Download and extract GloVe-6B if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(GLOVE_TXT):
        print(f"GloVe text file already exists at {GLOVE_TXT}")
        return
    if not os.path.exists(GLOVE_ZIP):
        print(f"Downloading GloVe-6B from {GLOVE_URL} ...")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
        print(f"Downloaded to {GLOVE_ZIP}")
    print("Extracting glove.6B.50d.txt ...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as zf:
        # Only extract the 50d file to save space
        zf.extract("glove.6B.50d.txt", DATA_DIR)
    print("Extraction complete.")


def load_glove_vectors():
    """Load GloVe vectors into a numpy array. Caches to .npy for fast reload."""
    if os.path.exists(GLOVE_NPY) and os.path.exists(GLOVE_WORDS):
        print("Loading cached GloVe vectors from .npy ...")
        vectors = np.load(GLOVE_NPY)
        words = np.load(GLOVE_WORDS, allow_pickle=True)
        print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
        return words, vectors

    print("Parsing GloVe text file ...")
    words = []
    vectors = []
    with open(GLOVE_TXT, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])

    words = np.array(words)
    vectors = np.array(vectors, dtype=np.float32)
    print(f"Parsed {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")

    # Cache for fast reloading
    np.save(GLOVE_NPY, vectors)
    np.save(GLOVE_WORDS, words)
    print("Cached to .npy files.")
    return words, vectors


def experiment1_varying_k(vectors):
    """
    Experiment 1: K-means with varying k.

    Measures how clustering objective, time, and empty cluster count
    change as the number of clusters increases.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: K-means with varying k on GloVe-6B-50d")
    print("=" * 70)

    n, d = vectors.shape
    k_values = [16, 64, 256, 1024, 4096]
    results = []

    for k in k_values:
        print(f"\n--- k = {k} ---")
        km = faiss.Kmeans(d, k, niter=25, verbose=False, seed=42)

        t0 = time.time()
        km.train(vectors)
        elapsed = time.time() - t0

        # Get iteration stats (list of dicts in pip-installed faiss-cpu)
        stats = km.iteration_stats
        objectives = [s["obj"] for s in stats]
        n_empty = stats[-1]["nsplit"]

        # Compute assignment stats
        _, assignments = km.assign(vectors)
        unique_clusters = len(np.unique(assignments))

        result = {
            "k": k,
            "final_obj": objectives[-1],
            "first_obj": objectives[0],
            "convergence_ratio": objectives[-1] / objectives[0],
            "time_s": elapsed,
            "n_empty_splits": n_empty,
            "unique_clusters_used": unique_clusters,
            "obj_per_iter": objectives,
        }
        results.append(result)

        print(f"  Final objective:     {objectives[-1]:.2f}")
        print(f"  First objective:     {objectives[0]:.2f}")
        print(f"  Convergence ratio:   {objectives[-1] / objectives[0]:.4f}")
        print(f"  Time:                {elapsed:.2f}s")
        print(f"  Empty splits (last): {n_empty}")
        print(f"  Unique clusters:     {unique_clusters} / {k}")

    # Summary table
    print("\n\nSUMMARY TABLE - Experiment 1")
    print("-" * 85)
    print(f"{'k':>6} | {'Final Obj':>12} | {'Conv. Ratio':>12} | {'Time (s)':>10} | {'Splits':>7} | {'Used/k':>10}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['k']:>6} | {r['final_obj']:>12.2f} | {r['convergence_ratio']:>12.4f} | "
            f"{r['time_s']:>10.2f} | {r['n_empty_splits']:>7} | "
            f"{r['unique_clusters_used']:>4}/{r['k']:<4}"
        )
    print("-" * 85)

    return results


def experiment2_niter_nredo(vectors):
    """
    Experiment 2: Effect of niter and nredo on clustering quality.

    Fixes k=256, varies the number of iterations and number of
    independent re-runs to study quality/cost tradeoffs.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Effect of niter and nredo (k=256) on GloVe-6B-50d")
    print("=" * 70)

    n, d = vectors.shape
    k = 256
    niter_values = [5, 10, 25, 50]
    nredo_values = [1, 3, 5]
    results = []

    for nredo in nredo_values:
        for niter in niter_values:
            print(f"\n--- niter={niter}, nredo={nredo} ---")
            km = faiss.Kmeans(d, k, niter=niter, nredo=nredo, verbose=False, seed=42)

            t0 = time.time()
            km.train(vectors)
            elapsed = time.time() - t0

            stats = km.iteration_stats
            final_obj = stats[-1]["obj"]

            result = {
                "niter": niter,
                "nredo": nredo,
                "final_obj": final_obj,
                "time_s": elapsed,
            }
            results.append(result)

            print(f"  Final objective: {final_obj:.2f}")
            print(f"  Time:            {elapsed:.2f}s")

    # Summary table
    print("\n\nSUMMARY TABLE - Experiment 2")
    print("-" * 70)
    print(f"{'nredo':>6} | {'niter':>6} | {'Final Obj':>12} | {'Time (s)':>10} | {'Obj Improvement':>16}")
    print("-" * 70)

    # Use the (niter=5, nredo=1) result as baseline
    baseline_obj = next(r["final_obj"] for r in results if r["niter"] == 5 and r["nredo"] == 1)
    for r in results:
        improvement = (baseline_obj - r["final_obj"]) / baseline_obj * 100
        print(
            f"{r['nredo']:>6} | {r['niter']:>6} | {r['final_obj']:>12.2f} | "
            f"{r['time_s']:>10.2f} | {improvement:>15.2f}%"
        )
    print("-" * 70)

    return results


def show_cluster_examples(words, vectors, k=64, n_examples=8):
    """Show example words from a few clusters for qualitative inspection."""
    print("\n" + "=" * 70)
    print(f"QUALITATIVE: Example words from k={k} clusters")
    print("=" * 70)

    n, d = vectors.shape
    km = faiss.Kmeans(d, k, niter=25, verbose=False, seed=42)
    km.train(vectors)
    _, assignments = km.assign(vectors)

    # Pick a few clusters with reasonable size to display
    cluster_ids, counts = np.unique(assignments, return_counts=True)
    # Sort by count descending, show top 5 and a few smaller ones
    sorted_idx = np.argsort(-counts)

    display_clusters = list(sorted_idx[:5]) + list(sorted_idx[-3:])
    for idx in display_clusters:
        cid = cluster_ids[idx]
        mask = assignments == cid
        cluster_words = words[mask]
        sample = cluster_words[:n_examples]
        print(f"\n  Cluster {cid:>3} ({counts[idx]:>5} words): {', '.join(sample)}")


def main():
    print("FAISS Clustering Experiments on GloVe Embeddings")
    print("=" * 70)

    # Download and load data
    download_glove()
    words, vectors = load_glove_vectors()

    n, d = vectors.shape
    print(f"\nDataset: GloVe-6B-50d — {n} words, {d} dimensions")
    print(f"Vector stats: mean={vectors.mean():.4f}, std={vectors.std():.4f}, "
          f"norm_mean={np.linalg.norm(vectors, axis=1).mean():.4f}")

    # Run experiments
    results1 = experiment1_varying_k(vectors)
    results2 = experiment2_niter_nredo(vectors)
    show_cluster_examples(words, vectors)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "clustering_glove_results.txt")
    with open(results_file, "w") as f:
        f.write("FAISS Clustering Experiments on GloVe-6B-50d\n")
        f.write("=" * 70 + "\n\n")

        f.write("EXPERIMENT 1: K-means with varying k\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'k':>6} | {'Final Obj':>12} | {'Conv. Ratio':>12} | {'Time (s)':>10} | {'Splits':>7} | {'Used/k':>10}\n")
        f.write("-" * 85 + "\n")
        for r in results1:
            f.write(
                f"{r['k']:>6} | {r['final_obj']:>12.2f} | {r['convergence_ratio']:>12.4f} | "
                f"{r['time_s']:>10.2f} | {r['n_empty_splits']:>7} | "
                f"{r['unique_clusters_used']:>4}/{r['k']:<4}\n"
            )
        f.write("-" * 85 + "\n\n")

        f.write("EXPERIMENT 2: Effect of niter and nredo (k=256)\n")
        f.write("-" * 70 + "\n")
        baseline_obj = next(r["final_obj"] for r in results2 if r["niter"] == 5 and r["nredo"] == 1)
        f.write(f"{'nredo':>6} | {'niter':>6} | {'Final Obj':>12} | {'Time (s)':>10} | {'Obj Improvement':>16}\n")
        f.write("-" * 70 + "\n")
        for r in results2:
            improvement = (baseline_obj - r["final_obj"]) / baseline_obj * 100
            f.write(
                f"{r['nredo']:>6} | {r['niter']:>6} | {r['final_obj']:>12.2f} | "
                f"{r['time_s']:>10.2f} | {improvement:>15.2f}%\n"
            )
        f.write("-" * 70 + "\n")

    print(f"\n\nResults saved to {results_file}")
    print("Done!")


if __name__ == "__main__":
    main()
