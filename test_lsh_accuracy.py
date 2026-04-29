"""
test_lsh_accuracy.py — Test LSH Index Accuracy (Phase 1)
=========================================================

Measures Precision@10 against brute-force ground truth.

Test configuration:
  - 100,000 random neurons (64-d, unit sphere)
  - 100 random queries
  - Clustered configuration (10 clusters) for harder test
  - Each query: compute brute-force top-10, compare with LSH top-10

Pass condition: Precision@10 > 90%

Usage:
  python3 test_lsh_accuracy.py
  python3 test_lsh_accuracy.py --fast          # 10K neurons, 20 queries
  python3 test_lsh_accuracy.py --clusters=10   # custom clusters
"""

import math
import random
import sys
import time
import os

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_lsh_index import LSHIndex, _NUMPY_AVAILABLE

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
DIMENSION = 64
TOP_K = 10
TARGET_PRECISION = 90.0  # target Precision@10 (%)
SEED = 42


def generate_clustered_data(
    n_neurons: int,
    n_clusters: int,
    spread: float = 0.3,
    seed: int = SEED,
):
    """
    Generate clustered random data.
    Each cluster has a random centroid, neurons are centroid + Gaussian noise.

    Returns:
        centroids: list of (centroid_vector, cluster_id)
        neurons: list of (neuron_id, vector)
    """
    rng = random.Random(seed)
    centroids = []

    for c in range(n_clusters):
        centroid = [rng.gauss(0, 1) for _ in range(DIMENSION)]
        norm = math.sqrt(sum(v * v for v in centroid))
        if norm > 0:
            centroid = [v / norm for v in centroid]
        centroids.append(centroid)

    chunksize = max(1, n_neurons // n_clusters)
    vectors = []
    for i in range(n_neurons):
        cluster = i % n_clusters
        centroid = centroids[cluster]
        vec = [c + rng.gauss(0, spread) for c in centroid]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        vectors.append((i, vec))

    return centroids, vectors


def brute_force_topk(
    query: list[float],
    vectors: list[tuple[int, list[float]]],
    top_k: int = TOP_K,
) -> set[int]:
    """Compute exact top-k via brute force cosine similarity."""
    qnorm = math.sqrt(sum(v * v for v in query))
    if qnorm == 0:
        return set()
    q = [v / qnorm for v in query]

    results = []
    for nid, vec in vectors:
        dot = sum(a * b for a, b in zip(q, vec))
        if len(results) < top_k:
            results.append((dot, nid))
            if len(results) == top_k:
                results.sort(reverse=True)
        elif dot > results[-1][0]:
            lo, hi = 0, top_k - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if results[mid][0] > dot:
                    lo = mid + 1
                else:
                    hi = mid
            results.insert(lo, (dot, nid))
            results.pop()

    return {nid for _, nid in results}


def run_test(
    n_neurons: int = 100_000,
    n_queries: int = 100,
    n_clusters: int = 10,
    probe_bits: int = 3,
    spread: float = 0.3,
    verbose: bool = True,
) -> dict:
    """
    Full accuracy test.

    Returns dict with:
      precision_avg, precision_min, precision_max
      latency_avg, latency_min, latency_max
      n_neurons, n_queries, n_clusters
      passed (bool)
    """
    import shutil

    if verbose:
        print(f"{'=' * 65}")
        print(f"  LSH ACCURACY TEST — Random Projections")
        print(f"{'=' * 65}")
        print(f"  Neurons:    {n_neurons:>10,}")
        print(f"  Queries:    {n_queries:>10,}")
        print(f"  Clusters:   {n_clusters:>10,}")
        print(f"  Spread:     {spread:>10.2f}")
        print(f"  Probe bits: {probe_bits:>10,}")
        print(f"  Mode:       {'numpy' if _NUMPY_AVAILABLE else 'pure Python'}")
        print()

    # ── 1. Generate data ──
    if verbose:
        print("  Generating data... ", end="", flush=True)

    centroids, neurons = generate_clustered_data(n_neurons, n_clusters, spread)
    # Generate query vectors from centroid + small noise
    rng = random.Random(SEED + 1)
    queries = []
    for _ in range(n_queries):
        cluster = rng.randint(0, n_clusters - 1)
        centroid = centroids[cluster]
        qv = [c + rng.gauss(0, spread * 0.5) for c in centroid]
        norm = math.sqrt(sum(v * v for v in qv))
        if norm > 0:
            qv = [v / norm for v in qv]
        queries.append(qv)

    if verbose:
        print(f"done ({n_neurons:,} neurons, {n_queries} queries)")

    # ── 2. Build LSH index ──
    d = os.path.expanduser("~/neuron-data/test_accuracy")
    if os.path.exists(d):
        shutil.rmtree(d)

    idx = LSHIndex(data_dir=d, use_numpy=_NUMPY_AVAILABLE)

    if verbose:
        print("  Building LSH index... ", end="", flush=True)

    t0 = time.perf_counter()
    for nid, vec in neurons:
        idx.add(nid, vec)
    idx.save_metadata()
    build_time = time.perf_counter() - t0

    if verbose:
        stats = idx.get_stats()
        print(f"done ({build_time:.2f}s)")
        print(f"  Tables:         {stats['tables']:,}")
        print(f"  Bits/table:     {stats['bits_per_table']}")
        print(f"  Buckets/table:  {stats['buckets_per_table']:,}")
        print()

    # ── 3. Query ──
    if verbose:
        print(f"  Running {n_queries} queries...")
        print(f"  {'#':>4s}  {'Lat(ms)':>8s}  {'Prec@10':>8s}  {'Cand':>6s}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}")

    all_vectors = list(zip(idx.neuron_ids, idx.vectors))

    precisions = []
    latencies = []
    candidate_counts = []

    for qi, qv in enumerate(queries):
        # Brute force ground truth
        brute_top = brute_force_topk(qv, all_vectors, TOP_K)

        # LSH search
        t0 = time.perf_counter()
        lsh_results = idx.search(qv, top_k=TOP_K)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        lsh_top = set(n for n, _ in lsh_results)
        overlap = brute_top & lsh_top
        prec = len(overlap) / TOP_K * 100
        precisions.append(prec)

        candidates = len(lsh_results) if lsh_results else 0
        candidate_counts.append(candidates)

        if verbose:
            status = "✓" if prec >= TARGET_PRECISION else " "
            print(f"  {qi:4d}  {ms:8.2f}  {prec:7.1f}%{status}  {candidates:6d}")

    # ── 4. Results ──
    avg_prec = sum(precisions) / len(precisions)
    min_prec = min(precisions)
    max_prec = max(precisions)
    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    avg_candidates = sum(candidate_counts) / len(candidate_counts)
    passed = avg_prec >= TARGET_PRECISION

    if verbose:
        print(f"\n  {'─' * 40}")
        print(f"  SUMMARY")
        print(f"  {'─' * 40}")
        print(f"  Precision@10  avg: {avg_prec:6.1f}%  min: {min_prec:5.1f}%  max: {max_prec:5.1f}%")
        print(f"  Latency       avg: {avg_lat:6.2f}ms  min: {min_lat:5.2f}ms  max: {max_lat:5.2f}ms")
        print(f"  Average candidates: {avg_candidates:.0f}")
        print(f"  Pass threshold:     > {TARGET_PRECISION:.0f}%")
        print(f"  Result:             {'✅ PASS' if passed else '❌ FAIL'}")
        print(f"  {'─' * 40}")

    return {
        "precision_avg": avg_prec,
        "precision_min": min_prec,
        "precision_max": max_prec,
        "latency_avg_ms": avg_lat,
        "latency_min_ms": min_lat,
        "latency_max_ms": max_lat,
        "candidates_avg": avg_candidates,
        "n_neurons": n_neurons,
        "n_queries": n_queries,
        "n_clusters": n_clusters,
        "build_time_s": build_time,
        "passed": passed,
    }


if __name__ == "__main__":
    n_neurons = 100_000
    n_queries = 100
    n_clusters = 10
    probe_bits = 3

    for arg in sys.argv[1:]:
        if arg == "--fast":
            n_neurons = 10_000
            n_queries = 20
            n_clusters = 5
        elif arg.startswith("--neurons="):
            n_neurons = int(arg.split("=")[1])
        elif arg.startswith("--queries="):
            n_queries = int(arg.split("=")[1])
        elif arg.startswith("--clusters="):
            n_clusters = int(arg.split("=")[1])
        elif arg.startswith("--probe="):
            probe_bits = int(arg.split("=")[1])
        elif arg == "--help":
            print("Usage: python3 test_lsh_accuracy.py [options]")
            print("  --fast              Quick test (10K neurons, 20 queries)")
            print("  --neurons=N         Neuron count (default: 100K)")
            print("  --queries=N         Query count (default: 100)")
            print("  --clusters=N        Cluster count (default: 10)")
            print("  --probe=N           Hamming neighbor probe bits (default: 3)")
            sys.exit(0)

    result = run_test(
        n_neurons=n_neurons,
        n_queries=n_queries,
        n_clusters=n_clusters,
        probe_bits=probe_bits,
    )

    sys.exit(0 if result["passed"] else 1)
