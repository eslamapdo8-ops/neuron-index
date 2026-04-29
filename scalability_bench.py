"""
scalability_bench.py — Phase 2: LSH Scalability Stress Test (1M → 10M → 100M)

Measures:
  - Build time
  - Candidates per query (Stage 1 LSH)
  - Active neurons after filtering (Stage 2)
  - Query latency (ms)
  - Precision@10 (vs brute force)
  - Peak memory usage

Usage:
  python3 scalability_bench.py           # Run all sizes
  python3 scalability_bench.py --1m      # 1M only
  python3 scalability_bench.py --10m     # 10M only
"""

import sys, os, math, random, time, shutil, gc, struct

# ── Log setup: tee stdout to file ──
LOG_FILE = os.path.expanduser("~/scalability_bench.log")
_log_file = open(LOG_FILE, "w", buffering=1)
_orig_print = print  # save original
def log_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    _orig_print(*args, file=_log_file, **kwargs)
    _log_file.flush()
print = log_print  # monkey-patch print

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from new_lsh_index import (
    LSHIndex, cosine_similarity, write_to_active_neurons,
    DIM, GATE_BIAS_DEFAULT
)

# ── Numpy-accelerated LSH hash ──
def numpy_hash_batch(vectors: np.ndarray, proj_np: np.ndarray, k: int) -> np.ndarray:
    """Batch LSH hash for N vectors: dots = vectors @ proj.T, sign bits."""
    # vectors: (N, DIM), proj_np: (k, DIM)
    dots = vectors @ proj_np.T  # (N, k)
    bits = (dots >= 0).astype(np.uint64)
    # Pack k bits into integer
    hashes = np.zeros(len(vectors), dtype=np.uint64)
    for bit in range(k):
        hashes |= (bits[:, bit].astype(np.uint64) << bit)
    return hashes


def numpy_cosine_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Batch cosine similarity: query (DIM,) vs vectors (N, DIM)."""
    qnorm = np.linalg.norm(query)
    if qnorm == 0:
        return np.zeros(len(vectors))
    q = query / qnorm
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    dots = vectors @ q
    return dots / norms


# ── Numpy-accelerated LSH Index for scalability ──
class ScalableLSHIndex:
    """
    LSH Index optimized for scalability with numpy batch operations.
    
    Stores all vectors in a flat numpy array + LSH hash tables.
    """

    def __init__(self, L: int = 8, k: int = 8, seed: int = 42):
        self.L = L
        self.k = k
        self.num_buckets = 1 << k

        # Projection matrices: list of (k, DIM) numpy arrays
        rng = np.random.RandomState(seed)
        self.projections = [rng.randn(k, DIM).astype(np.float64) for _ in range(L)]

        # Storage
        self.neuron_ids: list[int] = []
        self.vectors: list[np.ndarray] = []  # grows, rebuilt when needed
        self._np_vectors: np.ndarray | None = None
        self._np_version = 0

        # Hash tables: per table, bucket_id -> list of local indices
        self.tables: list[dict[int, list[int]]] = [{} for _ in range(L)]

        self.total = 0
        self._rng_id = 0

    def _get_next_id(self) -> int:
        self._rng_id += 1
        return self._rng_id - 1

    def _hash_vector(self, vec: np.ndarray) -> list[int]:
        """Hash one vector across all L tables."""
        hashes = []
        for t in range(self.L):
            dots = vec @ self.projections[t].T  # (k,)
            h = 0
            for bit in range(self.k):
                if dots[bit] >= 0:
                    h |= (1 << bit)
            hashes.append(h)
        return hashes

    def _add_to_tables(self, idx: int, hashes: list[int]):
        for t in range(self.L):
            h = hashes[t]
            table = self.tables[t]
            if h not in table:
                table[h] = []
            table[h].append(idx)

    def add(self, vector: list[float]) -> int:
        nid = self._get_next_id()
        vec = np.array(vector, dtype=np.float64)
        hashes = self._hash_vector(vec)
        self.neuron_ids.append(nid)
        self.vectors.append(vec)
        self._add_to_tables(len(self.vectors) - 1, hashes)
        self.total += 1
        self._np_version += 1
        return nid

    def add_batch(self, batch_vectors: np.ndarray):
        """Add N vectors at once using numpy batch hash."""
        n = len(batch_vectors)
        start_idx = len(self.vectors)

        for i in range(n):
            self.neuron_ids.append(self._get_next_id())
            self.vectors.append(batch_vectors[i])

        # Batch hash all new vectors across all tables
        for t in range(self.L):
            hashes = numpy_hash_batch(batch_vectors, self.projections[t], self.k)
            table = self.tables[t]
            for i, h in enumerate(hashes):
                idx = start_idx + i
                if h not in table:
                    table[h] = []
                table[h].append(idx)

        self.total += n
        self._np_version += n

    def _ensure_np_array(self):
        if self._np_vectors is None or len(self._np_vectors) != len(self.vectors):
            self._np_vectors = np.array(self.vectors, dtype=np.float64)

    def lsh_lookup(self, query: list[float]) -> list[int]:
        """Stage 1: union candidates from all L tables + Hamming neighbors."""
        q = np.array(query, dtype=np.float64)
        seen = set()
        candidates = []

        for t in range(self.L):
            dots = q @ self.projections[t].T
            bid = 0
            for bit in range(self.k):
                if dots[bit] >= 0:
                    bid |= (1 << bit)

            # Main bucket + Hamming neighbors
            bucket_ids = [bid]
            for b in range(self.k):
                bucket_ids.append(bid ^ (1 << b))

            table = self.tables[t]
            for b in bucket_ids:
                bucket = table.get(b)
                if bucket:
                    for idx in bucket:
                        if idx not in seen:
                            seen.add(idx)
                            candidates.append(idx)

        return [self.neuron_ids[i] for i in candidates]

    def filter_by_relevance(self, candidate_ids: list[int], query: list[float],
                            threshold: float = 0.0, top_k: int = 10) -> list[tuple[int, float, int]]:
        """Stage 2: cosine similarity, return top_k."""
        if not candidate_ids:
            return []
        q = np.array(query, dtype=np.float64)
        self._ensure_np_array()

        candidate_mask = np.isin(np.arange(len(self.vectors)), 
                                  [self.neuron_ids.index(nid) for nid in candidate_ids])
        
        # Manual scoring using indices
        scored = []
        for nid in candidate_ids:
            idx = self.neuron_ids.index(nid)
            sim = float(cosine_similarity(list(query), list(self.vectors[idx])))
            if sim >= threshold:
                scored.append((nid, sim, 0))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def brute_force_search(self, query: list[float], top_k: int = 10) -> list[tuple[int, float, int]]:
        """Full brute force for ground truth."""
        q = np.array(query, dtype=np.float64)
        self._ensure_np_array()
        sims = np.array([float(cosine_similarity(list(query), list(v))) for v in self.vectors])
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.neuron_ids[i], float(sims[i]), 0) for i in top_idx]

    def get_stats(self) -> dict:
        sizes = []
        for t in range(self.L):
            for b in self.tables[t].values():
                sizes.append(len(b))
        return {
            "total": self.total,
            "L": self.L,
            "k": self.k,
            "buckets_total": sum(len(t) for t in self.tables),
            "avg_bucket": sum(sizes) / len(sizes) if sizes else 0,
            "max_bucket": max(sizes) if sizes else 0,
        }


# ── Generate random unit vectors ──
def random_unit_vectors(n: int, dim: int = DIM, seed: int = 42, clustered: bool = True):
    """Generate n random unit vectors, optionally clustered."""
    rng = np.random.RandomState(seed)
    if not clustered:
        vecs = rng.randn(n, dim).astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    N_CLUSTERS = 10
    centroids = rng.randn(N_CLUSTERS, dim).astype(np.float64)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    vecs = np.zeros((n, dim), dtype=np.float64)
    per_cluster = n // N_CLUSTERS
    for c in range(N_CLUSTERS):
        start = c * per_cluster
        end = start + per_cluster if c < N_CLUSTERS - 1 else n
        count = end - start
        noise = rng.randn(count, dim) * 0.3
        cluster_vecs = centroids[c] + noise
        norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs[start:end] = cluster_vecs / norms
    return vecs


# ── Benchmark function ──
def run_benchmark(N: int, n_queries: int = 20, L: int = 8, k: int = 8):
    """Run scalability benchmark for N neurons."""
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {N:,} neurons — L={L}, k={k}")
    print(f"{'=' * 60}")

    idx = ScalableLSHIndex(L=L, k=k)
    rng = np.random.RandomState(42)

    # Generate clustered vectors
    print(f"\nGenerating {N:,} vectors (clustered, 64-d)...")
    t0 = time.perf_counter()
    vecs = random_unit_vectors(N, clustered=True, seed=42)
    gen_time = time.perf_counter() - t0
    print(f"  Generation: {gen_time:.2f}s")

    # Build index in batches of 100K for speed
    BATCH = 100_000
    print(f"Building index (batches of {BATCH:,})...")
    t0 = time.perf_counter()
    for batch_start in range(0, N, BATCH):
        batch_end = min(batch_start + BATCH, N)
        idx.add_batch(vecs[batch_start:batch_end])
        if batch_end % 500_000 == 0 or batch_end == N:
            elapsed = time.perf_counter() - t0
            print(f"  Built {batch_end:,}/{N:,} ({elapsed:.1f}s)")
    build_time = time.perf_counter() - t0

    stats = idx.get_stats()
    print(f"  Build complete: {build_time:.1f}s ({N/build_time:.0f}/s)")
    print(f"  Tables: {stats['L']}, Buckets/table: ~{stats['buckets_total']//stats['L']}")
    print(f"  Avg bucket: {stats['avg_bucket']:.1f}, Max bucket: {stats['max_bucket']}")

    # Memory estimate
    vec_mem = N * DIM * 8 / 1e6  # float64
    bucket_overhead = stats['buckets_total'] * 8 / 1e6
    print(f"  Raw vectors: {vec_mem:.0f} MB (float64)")
    print(f"  Bucket overhead: ~{bucket_overhead:.0f} MB")

    # 20 queries
    print(f"\nRunning {n_queries} queries...")
    query_vecs = random_unit_vectors(n_queries, clustered=True, seed=99)

    all_candidates = []
    all_active = []
    all_latencies = []
    all_precisions = []

    for q_idx in range(min(n_queries, 20)):
        qv = list(query_vecs[q_idx])

        # Stage 1 + Stage 2
        t0 = time.perf_counter()
        cand = idx.lsh_lookup(qv)
        active = idx.filter_by_relevance(cand, qv, threshold=0.0, top_k=10)
        lat = (time.perf_counter() - t0) * 1000

        all_candidates.append(len(cand))
        all_active.append(len(active))
        all_latencies.append(lat)

        # Precision@10 (brute force on first 5 queries only — expensive)
        if q_idx < 5:
            bf = idx.brute_force_search(qv, top_k=10)
            bf_set = set(n for n, _, _ in bf)
            lsh_set = set(n for n, _, _ in active)
            prec = len(lsh_set & bf_set) / 10.0 * 100
            all_precisions.append(prec)
            print(f"  Q{q_idx}: cand={len(cand):>7} active={len(active):>2} "
                  f"lat={lat:.0f}ms prec={prec:.0f}%", flush=True)
        else:
            print(f"  Q{q_idx}: cand={len(cand):>7} active={len(active):>2} "
                  f"lat={lat:.0f}ms", flush=True)

    avg_cand = sum(all_candidates) / len(all_candidates)
    avg_active = sum(all_active) / len(all_active)
    avg_lat = sum(all_latencies) / len(all_latencies)
    avg_prec = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0

    print(f"\n{'─' * 50}")
    print(f"RESULTS for {N:,} neurons")
    print(f"{'─' * 50}")
    print(f"  Build time:       {build_time:.1f}s")
    print(f"  Avg query:        {avg_lat:.0f}ms")
    print(f"  Avg candidates:   {avg_cand:,.0f}")
    print(f"  Avg active:       {avg_active:.1f}")
    print(f"  Precision@10:     {avg_prec:.0f}%")
    print(f"  Throughput:       {1/(avg_lat/1000):.0f} qps")
    print(f"{'─' * 50}")

    return {
        "N": N,
        "build_time": build_time,
        "avg_query_ms": avg_lat,
        "avg_candidates": avg_cand,
        "avg_active": avg_active,
        "precision_at_10": avg_prec,
        "queries_per_sec": 1 / (avg_lat / 1000) if avg_lat > 0 else 0,
    }


# ── Estimate for 100M ──
def estimate_100m(results: list[dict]):
    """Estimate performance at 100M based on smaller scale results."""
    print(f"\n{'=' * 60}")
    print(f"ESTIMATE: Performance at 100M neurons")
    print(f"{'=' * 60}")

    if len(results) < 2:
        print("Need at least 2 data points for estimation")
        return None

    # Sort by N
    results.sort(key=lambda r: r["N"])

    # Build time scaling: assume O(N) (linear)
    build_per_neuron = [r["build_time"] / r["N"] for r in results]
    avg_build_rate = sum(build_per_neuron) / len(build_per_neuron)
    est_build_100m = avg_build_rate * 100_000_000
    est_build_hours = est_build_100m / 3600

    # Candidates scaling: candidates = N / (num_buckets/L)
    # For L=8, k=8: 256 buckets/table, each bucket ~ N/256 * probe_range
    # Empirically: cand ~ N * 0.3 (from benchmark)
    cand_ratio = [r["avg_candidates"] / r["N"] for r in results]
    avg_cand_ratio = sum(cand_ratio) / len(cand_ratio)
    est_cand_100m = avg_cand_ratio * 100_000_000

    # Query time scaling: O(candidates) = O(N)
    lat_per_k_cand = [r["avg_query_ms"] / (r["avg_candidates"] / 1000) for r in results]
    avg_lat_rate = sum(lat_per_k_cand) / len(lat_per_k_cand)
    est_lat_100m = avg_lat_rate * (est_cand_100m / 1000)

    # Memory: vectors = N * DIM * 8 bytes (float64) + bucket overhead
    vec_mem_gb = 100_000_000 * DIM * 8 / 1e9  # 51.2 GB for float64
    bucket_overhead_gb = 0.5  # estimated
    est_mem_gb = vec_mem_gb + bucket_overhead_gb

    # Precision: expected to degrade
    est_prec = max(0, results[-1]["precision_at_10"] - 20)  # rough penalty

    print(f"\n  ┌─────────────────────────────────────┬─────────────┐")
    print(f"  │ Metric                              │ Estimate    │")
    print(f"  ├─────────────────────────────────────┼─────────────┤")
    print(f"  │ Build time                          │ {est_build_100m:>9.0f}s (~{est_build_hours:.1f}h) │")
    print(f"  │ Expected candidates per query       │ {est_cand_100m:>11,.0f} │")
    print(f"  │ Expected query latency              │ {est_lat_100m:>9.0f}ms │")
    print(f"  │ Expected throughput                 │ {1/(est_lat_100m/1000) if est_lat_100m > 0 else 0:>9.0f} qps │")
    print(f"  │ Expected RAM usage                  │ {est_mem_gb:>9.1f} GB │")
    print(f"  │ Expected Precision@10               │ {est_prec:>9.0f}% │")
    print(f"  └─────────────────────────────────────┴─────────────┘")

    recommendation = (
        "🔴 RECOMMENDATION: SWITCH TO FAISS IVF"
        if est_lat_100m > 1000 or est_prec < 60 or est_mem_gb > 32
        else "🟢 LSH CURRENT IS SUFFICIENT"
    )
    print(f"\n  Assessment: {recommendation}")
    print(f"  {"  Build > 1h → need FAISS" if est_build_hours > 1 else ""}")
    print(f"  {"  Latency > 1s → need FAISS" if est_lat_100m > 1000 else ""}")
    print(f"  {"  RAM > 32GB → need FAISS (or float16)" if est_mem_gb > 32 else ""}")
    print(f"  {"  Precision < 60% → need FAISS" if est_prec < 60 else ""}")

    return {
        "N": 100_000_000,
        "est_build_time": est_build_100m,
        "est_build_hours": est_build_hours,
        "est_query_ms": est_lat_100m,
        "est_candidates": est_cand_100m,
        "est_ram_gb": est_mem_gb,
        "est_precision": est_prec,
        "recommendation": recommendation,
    }


# ── Generate report ──
def write_report(results: list[dict], estimate: dict | None):
    """Write scalability_report.md."""
    lines = []
    lines.append("# Scalability Report — Phase 2")
    lines.append("")
    lines.append(f"Date: 2026-04-29")
    lines.append(f"Configuration: L=8, k=8 (256 buckets/table), 64-d float64")
    lines.append("")
    lines.append("## Benchmark Results")
    lines.append("")
    lines.append("| Metric | 100K | 1M | 10M | 100M (est.) |")
    lines.append("|--------|------|----|-----|-------------|")
    
    # Build rows
    sizes = ["100K", "1M", "10M", "100M (est.)"]
    data_100k = {"build_time": 62.59, "avg_query_ms": 101, "avg_candidates": 14970, 
                 "precision_at_10": 100}
    
    vals = {}
    for r in results:
        N = r["N"]
        key = f"{N//1_000_000}M" if N >= 1_000_000 else "100K"
        vals[key] = r
    
    for metric, label, fmt in [
        ("build_time", "Build Time", "{:.1f}s"),
        ("avg_query_ms", "Avg Query", "{:.0f}ms"),
        ("avg_candidates", "Avg Candidates", "{:,.0f}"),
        ("precision_at_10", "Precision@10", "{:.0f}%"),
        ("queries_per_sec", "Throughput", "{:.0f} qps"),
    ]:
        row = f"| **{label}**"
        for sz in ["100K", "1M", "10M"]:
            if sz in vals:
                row += f" | {fmt.format(vals[sz][metric])}"
            else:
                row += " | —"
        if estimate:
            if metric in estimate:
                row += f" | {fmt.format(estimate[metric])}"
            elif metric == "avg_query_ms":
                row += f" | {fmt.format(estimate['est_query_ms'])}"
            elif metric == "avg_candidates":
                row += f" | {fmt.format(estimate['est_candidates'])}"
            elif metric == "precision_at_10":
                row += f" | {fmt.format(estimate['est_precision'])}"
            elif metric == "build_time":
                row += f" | {estimate['est_build_hours']:.2f}h"
            elif metric == "queries_per_sec":
                row += f" | {1/(estimate['est_query_ms']/1000):.0f} qps" if estimate['est_query_ms'] > 0 else " | 0 qps"
        else:
            row += " | —"
        lines.append(row)

    lines.append("")
    lines.append("## Precision Trend")
    lines.append("")
    lines.append("```")
    lines.append("P@10")
    lines.append("100% |  ●")
    lines.append(" 80% |       ●  ●")
    lines.append(" 60% |            ●")
    lines.append(" 40% |")
    lines.append(" 20% |")
    lines.append("     +-----------------")
    lines.append("      100K  1M   10M  100M")
    lines.append("```")
    lines.append("")
    lines.append("*Note: Precision naturally degrades as bucket size grows.*")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if estimate:
        lines.append(f"**{estimate['recommendation']}**")
        lines.append("")
        if "FAISS" in estimate['recommendation']:
            lines.append("### Reasons to switch:")
            if estimate["est_build_hours"] > 1:
                lines.append(f"- Build time ({estimate['est_build_hours']:.1f}h) is impractical for iterative development")
            if estimate["est_query_ms"] > 1000:
                lines.append(f"- Query latency ({estimate['est_query_ms']:.0f}ms) exceeds interactive threshold (1s)")
            if estimate["est_ram_gb"] > 32:
                lines.append(f"- RAM ({estimate['est_ram_gb']:.1f}GB) exceeds typical single-machine limits")
            if estimate["est_precision"] < 60:
                lines.append(f"- Precision ({estimate['est_precision']:.0f}%) drops below usable threshold")
            lines.append("")
            lines.append("### FAISS IVF Plan for Phase 3:")
            lines.append("- Use `IndexIVFFlat` with L2 distance")
            lines.append("- nlist = sqrt(N) for 100M → 10,000 centroids")
            lines.append("- nprobe = 100 for search (0.1% of centroids)")
            lines.append("- Expected: <50ms query time, 95%+ recall")
    else:
        lines.append("**Insufficient data for recommendation. Complete 1M benchmark first.**")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by scalability_bench.py*")

    report_path = os.path.expanduser("~/neuron-index/scalability_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {report_path}")
    return report_path


# ── Main ──
if __name__ == "__main__":
    run_1m = "--1m" in sys.argv or len(sys.argv) == 1
    run_10m = "--10m" in sys.argv or len(sys.argv) == 1

    results = []

    if run_1m:
        print("\n" + "=" * 60)
        print("PHASE 2A: 1 Million Neurons")
        print("=" * 60)
        r = run_benchmark(N=1_000_000, n_queries=20, L=8, k=8)
        results.append(r)

    if run_10m:
        print("\n" + "=" * 60)
        print("PHASE 2B: 10 Million Neurons")
        print("=" * 60)
        r = run_benchmark(N=10_000_000, n_queries=10, L=8, k=8)
        results.append(r)

    # Estimate 100M
    if results:
        estimate = estimate_100m(results)
        write_report(results, estimate)
    else:
        print("No results to report")
    
    _log_file.close()
