"""
Experiment: HNSW Graph Node Reordering for Disk Block Locality

For disk-based graph indices, reordering nodes to improve neighbor locality
in disk blocks reduces the number of distinct blocks touched during search.

This script:
1. Builds an HNSW32,SQ8 index on a dataset
2. Evaluates baseline recall
3. Extracts visited node sets via Python-level BFS mirroring HNSW search
4. Implements several reordering strategies
5. Evaluates block locality (distinct blocks = id // BLOCK_SIZE)
"""

import numpy as np
import time
import sys
import heapq

import faiss
from faiss.contrib.datasets import SyntheticDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMENSION = 100
N_DATABASE = 200_000       # database size
N_QUERIES = 1000           # number of queries
N_VISITED_QUERIES = 200    # queries for visited-node analysis (Python BFS is slower)
N_TRAIN = 50_000           # training vectors for SQ8
HNSW_M = 32               # HNSW connectivity parameter
EF_CONSTRUCTION = 40       # build-time expansion factor
EF_SEARCH = 64             # search-time expansion factor
K = 10                     # top-k neighbors to retrieve
BLOCK_SIZE = 16            # disk block holds this many node records
SEED = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset():
    """Load dataset. Uses SyntheticDataset (clustered Gaussian) which has
    realistic locality structure."""
    print(f"Generating synthetic dataset: d={DIMENSION}, nb={N_DATABASE}, "
          f"nq={N_QUERIES}, nt={N_TRAIN}")
    ds = SyntheticDataset(DIMENSION, N_TRAIN, N_DATABASE, N_QUERIES, seed=SEED)
    xb = ds.get_database()
    xq = ds.get_queries()
    xt = ds.get_train()
    gt = ds.get_groundtruth(k=K)
    print(f"  xb: {xb.shape}, xq: {xq.shape}, xt: {xt.shape}, gt: {gt.shape}")
    return xb, xq, xt, gt

# ---------------------------------------------------------------------------
# Index Building & Recall Evaluation
# ---------------------------------------------------------------------------

def build_index(xb, xt):
    """Build HNSW32,SQ8 index."""
    print(f"\nBuilding HNSW{HNSW_M},SQ8 index...")
    t0 = time.time()
    index = faiss.index_factory(DIMENSION, f"HNSW{HNSW_M},SQ8")
    index.train(xt)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(xb)
    t1 = time.time()
    print(f"  Built in {t1-t0:.1f}s, ntotal={index.ntotal}")
    return index


def evaluate_recall(index, xq, gt):
    """Evaluate recall@K using faiss.eval_intersection."""
    index.hnsw.efSearch = EF_SEARCH
    t0 = time.time()
    D, I = index.search(xq, K)
    t1 = time.time()

    # eval_intersection returns total number of matching entries
    ninter = faiss.eval_intersection(I, gt[:, :K])
    recall = ninter / (len(xq) * K)
    print(f"\nRecall@{K} (efSearch={EF_SEARCH}): {recall:.4f}")
    print(f"  Search time: {(t1-t0)*1000:.1f}ms total, "
          f"{(t1-t0)*1000/len(xq):.2f}ms/query")

    # Also show ndis stats
    faiss.cvar.hnsw_stats.reset()
    index.search(xq, K)
    stats = faiss.cvar.hnsw_stats
    print(f"  Avg distances/query: {stats.ndis / len(xq):.0f}, "
          f"avg hops/query: {stats.nhops / len(xq):.1f}")
    return recall

# ---------------------------------------------------------------------------
# HNSW Graph Extraction
# ---------------------------------------------------------------------------

class HNSWGraph:
    """Extract and cache the HNSW graph structure for fast Python access."""

    def __init__(self, index):
        hnsw = index.hnsw
        self.ntotal = index.ntotal
        self.neighbors_arr = faiss.vector_to_array(hnsw.neighbors)
        self.offsets_arr = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
        self.levels_arr = faiss.vector_to_array(hnsw.levels)
        self.cum_nb = faiss.vector_to_array(hnsw.cum_nneighbor_per_level)
        self.entry_point = hnsw.entry_point
        self.max_level = hnsw.max_level
        self.efSearch = hnsw.efSearch

        # Precompute level-0 neighbor slices for fast access
        self._precompute_level0_neighbors()

    def _precompute_level0_neighbors(self):
        """Precompute list of level-0 neighbors for each node."""
        self.l0_neighbors = []
        cum0 = int(self.cum_nb[0])
        cum1 = int(self.cum_nb[1])
        max_degree = cum1 - cum0  # e.g. 2*M = 64 for M=32
        for i in range(self.ntotal):
            begin = int(self.offsets_arr[i]) + cum0
            end = int(self.offsets_arr[i]) + cum1
            nbrs = self.neighbors_arr[begin:end]
            valid = nbrs[nbrs >= 0]
            self.l0_neighbors.append(valid)

        # Also build a padded neighbor matrix for vectorized operations
        # Shape: (ntotal, max_degree), padded with -1
        self.l0_neighbor_matrix = np.full(
            (self.ntotal, max_degree), -1, dtype=np.int64)
        self.l0_degree = np.zeros(self.ntotal, dtype=np.int32)
        for i in range(self.ntotal):
            deg = len(self.l0_neighbors[i])
            self.l0_degree[i] = deg
            self.l0_neighbor_matrix[i, :deg] = self.l0_neighbors[i]

    def get_neighbors(self, node_id, level=0):
        """Get neighbors of a node at a given level."""
        if level == 0:
            return self.l0_neighbors[node_id]
        begin = int(self.offsets_arr[node_id]) + int(self.cum_nb[level])
        end = int(self.offsets_arr[node_id]) + int(self.cum_nb[level + 1])
        nbrs = self.neighbors_arr[begin:end]
        return nbrs[nbrs >= 0]

    def get_level(self, node_id):
        return self.levels_arr[node_id]

# ---------------------------------------------------------------------------
# Python-level HNSW Search to Collect Visited Nodes
# ---------------------------------------------------------------------------

def hnsw_search_visited(graph, query_vec, xb, ef_search, k):
    """
    Perform HNSW search in Python, collecting all visited node IDs.

    Mirrors the C++ HNSW::search logic:
    1. Greedy descent on upper levels to find entry point for level 0
    2. BFS-like expansion on level 0 with efSearch budget

    Returns: (result_ids, visited_set)
    """
    visited = set()

    def dist(node_id):
        """L2 distance between query and node."""
        diff = query_vec - xb[node_id]
        return float(np.dot(diff, diff))

    # Phase 1: greedy search on upper levels
    nearest = graph.entry_point
    d_nearest = dist(nearest)
    visited.add(nearest)

    for level in range(graph.max_level, 0, -1):
        changed = True
        while changed:
            changed = False
            for nbr in graph.get_neighbors(nearest, level):
                nbr = int(nbr)
                visited.add(nbr)
                d = dist(nbr)
                if d < d_nearest:
                    d_nearest = d
                    nearest = nbr
                    changed = True

    # Phase 2: search on level 0 with efSearch expansion
    # Use a min-heap for candidates (sorted by distance, closest first)
    # and a max-heap for results (sorted by distance, farthest first)
    ef = max(ef_search, k)

    # candidates: min-heap of (dist, node_id) - nodes to explore
    candidates = [(d_nearest, nearest)]
    # results: max-heap of (-dist, node_id) - best results so far
    results = [(-d_nearest, nearest)]
    visited.add(nearest)

    while candidates:
        d_cand, v_cand = heapq.heappop(candidates)

        # Check stopping: if candidate is worse than worst result and
        # we have enough results
        if len(results) >= ef and d_cand > -results[0][0]:
            break

        for nbr in graph.get_neighbors(v_cand, 0):
            nbr = int(nbr)
            if nbr in visited:
                continue
            visited.add(nbr)

            d_nbr = dist(nbr)

            if len(results) < ef or d_nbr < -results[0][0]:
                heapq.heappush(candidates, (d_nbr, nbr))
                heapq.heappush(results, (-d_nbr, nbr))
                if len(results) > ef:
                    heapq.heappop(results)

    # Extract top-k from results
    results.sort(key=lambda x: -x[0])  # sort by distance ascending
    result_ids = [r[1] for r in results[:k]]

    return result_ids, visited


def collect_visited_nodes(graph, xq, xb, ef_search, k, max_queries=None):
    """Run search for all queries, collecting visited node sets."""
    nq = len(xq)
    if max_queries:
        nq = min(nq, max_queries)

    print(f"\nCollecting visited nodes for {nq} queries (efSearch={ef_search})...")
    all_visited = []
    t0 = time.time()

    for i in range(nq):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{nq} queries ({elapsed:.1f}s)")
        _, visited = hnsw_search_visited(graph, xq[i], xb, ef_search, k)
        all_visited.append(visited)

    t1 = time.time()
    sizes = [len(v) for v in all_visited]
    print(f"  Done in {t1-t0:.1f}s")
    print(f"  Visited nodes per query: "
          f"mean={np.mean(sizes):.0f}, median={np.median(sizes):.0f}, "
          f"min={np.min(sizes)}, max={np.max(sizes)}")

    return all_visited

# ---------------------------------------------------------------------------
# Block Locality Evaluation
# ---------------------------------------------------------------------------

def count_blocks_touched(visited_nodes, indirection, block_size=BLOCK_SIZE):
    """Given a set of visited node IDs and an indirection table,
    count the number of distinct disk blocks touched.

    indirection[old_id] = new_id
    block = new_id // block_size
    """
    blocks = set()
    for node in visited_nodes:
        new_id = indirection[node]
        blocks.add(new_id // block_size)
    return len(blocks)


def evaluate_reordering(name, indirection, all_visited, block_size=BLOCK_SIZE):
    """Evaluate a reordering strategy across all queries."""
    block_counts = [
        count_blocks_touched(visited, indirection, block_size)
        for visited in all_visited
    ]
    mean_blocks = np.mean(block_counts)
    median_blocks = np.median(block_counts)
    p90_blocks = np.percentile(block_counts, 90)
    p99_blocks = np.percentile(block_counts, 99)

    # Also compute: what fraction of each block is used on average?
    # (measures how "full" each touched block is)
    utilizations = []
    for visited in all_visited:
        block_to_count = {}
        for node in visited:
            b = indirection[node] // block_size
            block_to_count[b] = block_to_count.get(b, 0) + 1
        if block_to_count:
            utilizations.append(np.mean(list(block_to_count.values())) / block_size)

    mean_util = np.mean(utilizations)

    print(f"  {name:40s}: mean={mean_blocks:7.1f}  median={median_blocks:7.1f}  "
          f"p90={p90_blocks:7.1f}  p99={p99_blocks:7.1f}  "
          f"block_util={mean_util:.3f}")
    return mean_blocks

# ---------------------------------------------------------------------------
# Reordering Strategies
# ---------------------------------------------------------------------------

def identity_reordering(ntotal):
    """Baseline: no reordering."""
    return np.arange(ntotal, dtype=np.int64)


def random_reordering(ntotal, seed=123):
    """Random permutation (control baseline)."""
    rng = np.random.RandomState(seed)
    perm = np.arange(ntotal, dtype=np.int64)
    rng.shuffle(perm)
    return perm


def lloyds_1d_reordering(graph, n_iters=20, alpha=0.2, n_neighbors=4):
    """
    1D Lloyd's algorithm for graph node reordering (vectorized).

    1. Initialize ranking r[i] = i
    2. Each iteration: r[i] = lerp(r[i], avg(r[n] for n in first-k-neighbors), alpha)
    3. Stable sort by r values
    4. Repeat n_iters times

    The indirection table maps old_id -> new_id.
    """
    ntotal = graph.ntotal
    r = np.arange(ntotal, dtype=np.float64)

    # Build truncated neighbor matrix: (ntotal, n_neighbors), padded with self-index
    nbr_mat = np.tile(np.arange(ntotal).reshape(-1, 1), (1, n_neighbors))
    nbr_count = np.zeros(ntotal, dtype=np.float64)

    for i in range(ntotal):
        nbrs = graph.l0_neighbors[i]
        k = min(len(nbrs), n_neighbors)
        if k > 0:
            nbr_mat[i, :k] = nbrs[:k]
            nbr_count[i] = k
        else:
            nbr_count[i] = 0

    # Nodes with no neighbors keep their rank
    has_neighbors = nbr_count > 0

    for iteration in range(n_iters):
        # Gather neighbor r values: shape (ntotal, n_neighbors)
        nbr_r = r[nbr_mat]
        # Average across neighbors (only count valid ones)
        # For nodes with fewer than n_neighbors, extra slots have self-index
        # which is fine as a fallback
        avg_r = np.where(
            has_neighbors.reshape(-1, 1),
            nbr_r,
            r.reshape(-1, 1)
        ).mean(axis=1)
        # Lerp
        r = np.where(has_neighbors, r * (1 - alpha) + avg_r * alpha, r)

    # Stable sort by r
    sorted_indices = np.argsort(r, kind='stable').astype(np.int64)
    indirection = np.empty(ntotal, dtype=np.int64)
    indirection[sorted_indices] = np.arange(ntotal)
    return indirection


def bfs_reordering(graph):
    """
    BFS (Cuthill-McKee style) reordering.

    Traverse the graph in BFS order starting from a random node.
    Nodes visited consecutively in BFS are likely to be accessed together
    during search, so placing them in consecutive positions improves locality.
    """
    ntotal = graph.ntotal
    visited = np.zeros(ntotal, dtype=bool)
    order = []

    # Start BFS from the HNSW entry point
    start = graph.entry_point
    queue = [start]
    visited[start] = True

    while queue:
        node = queue.pop(0)
        order.append(node)
        # Add unvisited neighbors
        for nbr in graph.l0_neighbors[node]:
            nbr = int(nbr)
            if not visited[nbr]:
                visited[nbr] = True
                queue.append(nbr)

    # Handle any disconnected components
    for i in range(ntotal):
        if not visited[i]:
            order.append(i)

    order = np.array(order, dtype=np.int64)
    # order[new_pos] = old_id
    indirection = np.empty(ntotal, dtype=np.int64)
    indirection[order] = np.arange(ntotal)
    return indirection


def spectral_1d_reordering(graph, seed=42):
    """
    Spectral ordering via the Fiedler vector of the graph Laplacian (vectorized).

    The Fiedler vector (2nd smallest eigenvector of the Laplacian) provides
    an optimal 1D embedding that minimizes the sum of squared differences
    between connected nodes. Sorting by this vector groups heavily-connected
    nodes together.

    Approximated via power iteration on the random-walk normalized adjacency
    matrix, deflating out the trivial constant eigenvector.
    """
    ntotal = graph.ntotal
    rng = np.random.RandomState(seed)

    # Use padded neighbor matrix for vectorized sparse mat-vec
    nbr_mat = graph.l0_neighbor_matrix  # (ntotal, max_degree), -1 padded
    degrees = graph.l0_degree.astype(np.float64)
    degrees_safe = np.maximum(degrees, 1)

    # Initialize with random values
    x = rng.randn(ntotal)
    x -= np.mean(x)

    n_iters = 50
    for it in range(n_iters):
        # Sparse mat-vec: new_x[i] = sum(x[neighbors[i]]) / degree[i]
        # Handle -1 padding by using x extended with a 0 sentinel
        x_ext = np.append(x, 0.0)  # index -1 maps to 0.0
        nbr_vals = x_ext[nbr_mat]  # (ntotal, max_degree)
        new_x = nbr_vals.sum(axis=1) / degrees_safe

        # For isolated nodes, keep their value
        isolated = degrees == 0
        new_x[isolated] = x[isolated]

        # Remove mean (deflate constant eigenvector)
        new_x -= np.mean(new_x)
        # Normalize
        norm = np.linalg.norm(new_x)
        if norm > 0:
            new_x /= norm
        x = new_x

    sorted_indices = np.argsort(x, kind='stable').astype(np.int64)
    indirection = np.empty(ntotal, dtype=np.int64)
    indirection[sorted_indices] = np.arange(ntotal)
    return indirection


def hilbert_reordering(xb):
    """
    Hilbert curve reordering based on vector coordinates.

    Maps high-dimensional vectors to 1D Hilbert curve positions, then sorts.
    Vectors close in the original space tend to be close on the Hilbert curve,
    which means their graph neighbors (which are also spatially close) will
    have nearby IDs.

    We approximate by using a PCA projection to 2D, then computing Hilbert
    indices on the 2D coordinates.
    """
    ntotal = xb.shape[0]
    d = xb.shape[1]

    # PCA to 2D for Hilbert mapping
    mean = xb.mean(axis=0)
    xc = xb - mean
    # Use a few random projections for efficiency
    # Actually, let's do proper PCA via SVD on a sample
    sample_size = min(ntotal, 10000)
    idx = np.random.choice(ntotal, sample_size, replace=False)
    cov_approx = xc[idx].T @ xc[idx] / sample_size

    # Get top 2 eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_approx)
    # eigh returns ascending order, take last 2
    proj = eigenvectors[:, -2:]  # (d, 2)

    coords_2d = xc @ proj  # (ntotal, 2)

    # Normalize to [0, 2^16) grid
    BITS = 16
    GRID = 2 ** BITS
    for dim in range(2):
        mn, mx = coords_2d[:, dim].min(), coords_2d[:, dim].max()
        coords_2d[:, dim] = (coords_2d[:, dim] - mn) / (mx - mn + 1e-10) * (GRID - 1)

    coords_int = coords_2d.astype(np.int64)
    coords_int = np.clip(coords_int, 0, GRID - 1)

    # Compute Hilbert index for each 2D point
    hilbert_indices = np.array([
        _xy_to_hilbert(int(coords_int[i, 0]), int(coords_int[i, 1]), BITS)
        for i in range(ntotal)
    ])

    sorted_indices = np.argsort(hilbert_indices, kind='stable').astype(np.int64)
    indirection = np.empty(ntotal, dtype=np.int64)
    indirection[sorted_indices] = np.arange(ntotal)
    return indirection


def _xy_to_hilbert(x, y, order):
    """Convert (x, y) to Hilbert curve index for a 2^order x 2^order grid."""
    d = 0
    s = (1 << order) >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def gp_reordering(graph, xb, n_clusters=None):
    """
    Graph Partitioning reordering via recursive k-means bisection.

    Recursively partition the vectors using k-means, assigning contiguous
    ID ranges to each partition. Vectors in the same cluster are spatially
    close and thus likely connected in the HNSW graph.
    """
    ntotal = xb.shape[0]
    if n_clusters is None:
        # Aim for clusters of about BLOCK_SIZE vectors
        n_clusters = max(1, ntotal // BLOCK_SIZE)

    # Use faiss k-means for efficiency
    d = xb.shape[1]
    niter = 20
    nclust = min(n_clusters, ntotal)

    kmeans = faiss.Kmeans(d, nclust, niter=niter, verbose=False, seed=SEED)
    kmeans.train(xb)

    # Assign each vector to nearest centroid
    _, assignments = kmeans.index.search(xb, 1)
    assignments = assignments.ravel()

    # Sort: first by cluster, then by original index within cluster (stable)
    order = np.argsort(assignments, kind='stable').astype(np.int64)
    indirection = np.empty(ntotal, dtype=np.int64)
    indirection[order] = np.arange(ntotal)
    return indirection


def combined_reordering(graph, xb, n_lloyd_iters=20, alpha=0.2):
    """
    Combined strategy: k-means clustering + 1D Lloyd's refinement (vectorized).

    First do coarse grouping via k-means (spatial locality), then refine
    within each group using 1D Lloyd's (graph connectivity locality).
    """
    ntotal = graph.ntotal
    n_neighbors = 4

    # Step 1: k-means partitioning to get coarse groups
    d = xb.shape[1]
    n_clusters = max(1, ntotal // (BLOCK_SIZE * 4))
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False, seed=SEED)
    kmeans.train(xb)
    _, assignments = kmeans.index.search(xb, 1)
    assignments = assignments.ravel()

    # Step 2: Initialize r based on cluster ordering
    cluster_order = np.argsort(assignments, kind='stable')
    r = np.zeros(ntotal, dtype=np.float64)
    r[cluster_order] = np.arange(ntotal, dtype=np.float64)

    # Build truncated neighbor matrix
    nbr_mat = np.tile(np.arange(ntotal).reshape(-1, 1), (1, n_neighbors))
    has_neighbors = np.zeros(ntotal, dtype=bool)
    for i in range(ntotal):
        nbrs = graph.l0_neighbors[i]
        k = min(len(nbrs), n_neighbors)
        if k > 0:
            nbr_mat[i, :k] = nbrs[:k]
            has_neighbors[i] = True

    # Step 3: Apply vectorized Lloyd's iterations
    for iteration in range(n_lloyd_iters):
        nbr_r = r[nbr_mat]
        avg_r = np.where(
            has_neighbors.reshape(-1, 1),
            nbr_r,
            r.reshape(-1, 1)
        ).mean(axis=1)
        r = np.where(has_neighbors, r * (1 - alpha) + avg_r * alpha, r)

    sorted_indices = np.argsort(r, kind='stable').astype(np.int64)
    indirection = np.empty(ntotal, dtype=np.int64)
    indirection[sorted_indices] = np.arange(ntotal)
    return indirection


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)

    # 1. Load data
    xb, xq, xt, gt = load_dataset()

    # 2. Build index
    index = build_index(xb, xt)

    # 3. Evaluate recall
    recall = evaluate_recall(index, xq, gt)

    # 4. Extract graph
    print("\nExtracting HNSW graph structure...")
    t0 = time.time()
    graph = HNSWGraph(index)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Print graph stats
    degrees = [len(graph.l0_neighbors[i]) for i in range(graph.ntotal)]
    print(f"  Level-0 degree: mean={np.mean(degrees):.1f}, "
          f"min={np.min(degrees)}, max={np.max(degrees)}")

    # 5. Collect visited nodes (use a subset of queries since Python BFS is slower)
    all_visited = collect_visited_nodes(
        graph, xq, xb, EF_SEARCH, K, max_queries=N_VISITED_QUERIES)

    # 6. Build reordering strategies
    ntotal = graph.ntotal
    print("\n" + "=" * 100)
    print("Building reordering strategies...")
    print("=" * 100)

    strategies = {}

    print("\n[1/7] Identity (baseline)...")
    strategies["Identity (baseline)"] = identity_reordering(ntotal)

    print("[2/7] Random permutation (control)...")
    strategies["Random permutation (control)"] = random_reordering(ntotal)

    print("[3/7] 1D Lloyd's (20 iters, alpha=0.2, 4 neighbors)...")
    t0 = time.time()
    strategies["1D Lloyd's (a=0.2, n=4, iters=20)"] = \
        lloyds_1d_reordering(graph, n_iters=20, alpha=0.2, n_neighbors=4)
    print(f"       ({time.time()-t0:.1f}s)")

    print("[4/7] BFS (Cuthill-McKee) ordering...")
    t0 = time.time()
    strategies["BFS (Cuthill-McKee)"] = bfs_reordering(graph)
    print(f"       ({time.time()-t0:.1f}s)")

    print("[5/7] Spectral (approx Fiedler vector)...")
    t0 = time.time()
    strategies["Spectral (approx Fiedler)"] = spectral_1d_reordering(graph)
    print(f"       ({time.time()-t0:.1f}s)")

    print("[6/7] Hilbert curve (PCA to 2D)...")
    t0 = time.time()
    strategies["Hilbert curve (PCA-2D)"] = hilbert_reordering(xb)
    print(f"       ({time.time()-t0:.1f}s)")

    print("[7/7] K-means partitioning...")
    t0 = time.time()
    strategies["K-means partitioning"] = gp_reordering(graph, xb)
    print(f"       ({time.time()-t0:.1f}s)")

    # Also test Lloyd's with different parameters
    print("\nLloyd's variants...")
    lloyd_params = [
        (0.1, 4, 20),
        (0.5, 4, 20),
        (0.2, 2, 20),
        (0.2, 8, 20),
        (0.2, 16, 20),
        (0.2, 4, 5),
        (0.2, 4, 50),
        (0.5, 16, 50),   # aggressive: high alpha, many neighbors, many iters
        (0.8, 32, 100),   # very aggressive
    ]
    for alpha, n_nbrs, n_iters in lloyd_params:
        t0 = time.time()
        name = f"1D Lloyd's (a={alpha}, n={n_nbrs}, iters={n_iters})"
        strategies[name] = lloyds_1d_reordering(
            graph, n_iters=n_iters, alpha=alpha, n_neighbors=n_nbrs)
        print(f"  {name}: {time.time()-t0:.1f}s")

    # Combined strategies
    print("\nCombined k-means + Lloyd's variants...")
    for alpha, n_iters in [(0.2, 20), (0.5, 50)]:
        t0 = time.time()
        name = f"K-means + Lloyd's (a={alpha}, iters={n_iters})"
        strategies[name] = combined_reordering(
            graph, xb, n_lloyd_iters=n_iters, alpha=alpha)
        print(f"  {name}: {time.time()-t0:.1f}s")

    # 7. Evaluate all strategies
    print("\n" + "=" * 100)
    print(f"Block Locality Results (block_size={BLOCK_SIZE})")
    print("=" * 100)
    print(f"  {'Strategy':40s}: {'mean':>7s}  {'median':>7s}  "
          f"{'p90':>7s}  {'p99':>7s}  {'blk_util':>8s}")
    print("-" * 100)

    results = {}
    baseline = None
    for name, indirection in strategies.items():
        mean_blocks = evaluate_reordering(name, indirection, all_visited)
        results[name] = mean_blocks
        if baseline is None:
            baseline = mean_blocks

    # 8. Summary with improvement ratios
    print("\n" + "=" * 100)
    print("Summary: Reduction in Mean Blocks Touched (vs Identity baseline)")
    print("=" * 100)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, mean_blocks in sorted_results:
        reduction = (1 - mean_blocks / baseline) * 100
        bar = "█" * max(0, int(reduction / 2))
        print(f"  {name:45s}: {mean_blocks:7.1f} blocks  "
              f"({reduction:+6.1f}%)  {bar}")

    # 9. Also evaluate at different block sizes
    print("\n" + "=" * 100)
    print("Sensitivity to Block Size")
    print("=" * 100)

    best_name = sorted_results[0][0]
    best_indirection = strategies[best_name]
    identity = strategies["Identity (baseline)"]

    for bs in [4, 8, 16, 32, 64, 128]:
        id_blocks = np.mean([
            count_blocks_touched(v, identity, bs) for v in all_visited])
        best_blocks = np.mean([
            count_blocks_touched(v, best_indirection, bs) for v in all_visited])
        reduction = (1 - best_blocks / id_blocks) * 100
        print(f"  block_size={bs:3d}: identity={id_blocks:7.1f}, "
              f"best({best_name[:25]})={best_blocks:7.1f}  "
              f"({reduction:+.1f}%)")


if __name__ == "__main__":
    main()
