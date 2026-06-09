# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
PCA-based 1-bit residual quantization experiment.

Each vector is encoded as 2*M bits via two stages:
  1. Rotate by PCA (no mean subtraction), quantize top M dims to sign bits
  2. Compute residual, rotate by second PCA, quantize top M dims again

Reconstruction uses significance = sqrt(2 * eigenvalue / pi), which is the
optimal reconstruction level for 1-bit quantization of a Gaussian.
"""

import argparse
import numpy as np
import faiss


def generate_data(d, n_train, n_test, seed=1234):
    rs = np.random.RandomState(seed)
    # decaying spectrum so PCA is meaningful
    U, _, _ = np.linalg.svd(rs.randn(d, d).astype("float32"))
    spectrum = 1.0 / np.arange(1, d + 1).astype("float32")
    cov_sqrt = U @ np.diag(np.sqrt(spectrum))
    x = (rs.randn(n_train + n_test, d) @ cov_sqrt.T).astype("float32")
    return x[:n_train], x[n_train:]


def train_pca_no_mean(x, d_in, M):
    pca = faiss.PCAMatrix(d_in, M, 0, False)
    pca.have_bias = False
    pca.train(x)
    eigenvalues = faiss.vector_to_array(pca.eigenvalues)
    return pca, eigenvalues


def quantize_reconstruct(pca, x_train, x, M):
    z_train = pca.apply_py(x_train)
    # second moment (not centered, matching have_bias=False)
    variance = np.mean(z_train ** 2, axis=0)

    z = pca.apply_py(x)
    bits = (z >= 0).astype(np.float32)
    significance = np.sqrt(2.0 * variance[:M] / np.pi)
    z_q = ((2.0 * bits - 1.0) * significance[np.newaxis, :]).astype("float32")
    x_partial = pca.reverse_transform(z_q)
    return x_partial, bits


def run_two_stage(x_train, x_test, d, M):
    pca1, _ = train_pca_no_mean(x_train, d, M)
    x_partial1_train, _ = quantize_reconstruct(pca1, x_train, x_train, M)
    x_partial1_test, bits1 = quantize_reconstruct(pca1, x_train, x_test, M)

    residual_train = x_train - x_partial1_train
    residual_test = x_test - x_partial1_test

    pca2, _ = train_pca_no_mean(residual_train, d, M)
    x_partial2_test, bits2 = quantize_reconstruct(
        pca2, residual_train, residual_test, M
    )

    return x_partial1_test + x_partial2_test, (bits1, bits2)


def run_single_stage(x_train, x_test, d, total_bits):
    pca, _ = train_pca_no_mean(x_train, d, total_bits)
    x_recon, bits = quantize_reconstruct(pca, x_train, x_test, total_bits)
    return x_recon


def run_pca_truncation(x_train, x_test, d, n_components):
    """Upper bound: keep exact PCA components (not bit-limited)."""
    pca = faiss.PCAMatrix(d, n_components, 0, False)
    pca.have_bias = False
    pca.train(x_train)
    z = pca.apply_py(x_test)
    return pca.reverse_transform(z)


def evaluate(x_original, x_reconstructed, label, n_pairs=50000):
    mse = np.mean(np.sum((x_original - x_reconstructed) ** 2, axis=1))
    energy = np.mean(np.sum(x_original ** 2, axis=1))
    rel_mse = mse / energy

    rs = np.random.RandomState(42)
    n = len(x_original)
    ia, ib = rs.randint(0, n, n_pairs), rs.randint(0, n, n_pairs)
    true_ip = np.sum(x_original[ia] * x_original[ib], axis=1)
    approx_ip = np.sum(x_reconstructed[ia] * x_reconstructed[ib], axis=1)
    ip_corr = np.corrcoef(true_ip, approx_ip)[0, 1]

    print(f"  {label:40s}  relMSE={rel_mse:.4f}  IP-corr={ip_corr:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--M", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    print(f"d={args.d}, n_train={args.n_train}, n_test={args.n_test}")
    x_train, x_test = generate_data(
        args.d, args.n_train, args.n_test, args.seed
    )

    for M in args.M:
        total_bits = 2 * M
        print(f"\n--- M={M}, total_bits={total_bits} ---")

        # two-stage residual (the experiment)
        x_recon, _ = run_two_stage(x_train, x_test, args.d, M)
        evaluate(x_test, x_recon, f"2-stage residual ({M}+{M} bits)")

        # single stage with same total bits
        x_recon = run_single_stage(x_train, x_test, args.d, total_bits)
        evaluate(x_test, x_recon, f"single-stage ({total_bits} bits)")

        # PCA truncation (float upper bound) at M components
        x_recon = run_pca_truncation(x_train, x_test, args.d, M)
        evaluate(x_test, x_recon, f"PCA truncation ({M} float dims)")

        # PCA truncation at 2M components
        x_recon = run_pca_truncation(x_train, x_test, args.d, total_bits)
        evaluate(x_test, x_recon, f"PCA truncation ({total_bits} float dims)")


if __name__ == "__main__":
    main()
