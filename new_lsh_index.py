"""
new_lsh_index.py — Multi-table LSH + Active Inference Neuron Index (Phase 1)
==============================================================================

Architecture:
  Layer 1 (LSH Multi-Table): L=8 independent LSH tables, each with k=12 bits.
    - Each table: random Gaussian projection matrix (k × 64)
    - 12-bit hash per table → 4,096 buckets per table
    - L tables = 8 × 4,096 = 32,768 logical buckets
    - Query: hash in all L tables, union candidates, deduplicate
    - Precision@10 > 90%

  Layer 2 (Filtering): 2-stage select_active_neurons:
    - Stage 1: lsh_lookup(query) → union of candidates from L tables
    - Stage 2: filter_by_relevance(candidates, threshold, gate_bias)

  CellState: 192-byte neuron memory (signature, context, links, gate_bias)

Dependencies: numpy (optional, for speed), pure Python fallback.
"""

import struct
import math
import random
import os
import json
import threading
import time
import sys

# ──────────────────────────────────────────────
# Numpy detection
# ──────────────────────────────────────────────
_NUMPY_AVAILABLE = False
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DIM = 64               # signature dimension
N_HASH_BITS = 12       # bits per LSH table → 4096 buckets
NUM_TABLES = 8         # L = 8 multi-table
NUM_BUCKETS = 1 << N_HASH_BITS  # 4096
GATE_BIAS_DEFAULT = 0.5
GATE_BIAS_MIN = 0.01
GATE_BIAS_MAX = 0.99


# ──────────────────────────────────────────────
# Cosine similarity
# ──────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ──────────────────────────────────────────────
# CellState — 192-byte neuron memory
# ──────────────────────────────────────────────
class CellState:
    """
    Internal state of a single neuron (metadata only; signature stored in index).
    """
    __slots__ = ("neuron_id", "signature", "context", "links",
                 "gate_bias", "access_count", "memory_version", "last_accessed")

    def __init__(self, neuron_id: int, signature: list[float],
                 context: bytes = b"", links: list[int] | None = None):
        self.neuron_id = neuron_id
        self.signature = list(signature)  # 64 floats
        # Context buffer: 64 bytes
        ctx = context if isinstance(context, bytes) else b""
        self.context = (ctx + b"\x00" * 64)[:64]
        # Link table: max 16 IDs
        self.links = (links or [])[:16]
        self.gate_bias = GATE_BIAS_DEFAULT
        self.access_count = 0
        self.memory_version = 0
        self.last_accessed = 0.0


# ──────────────────────────────────────────────
# LSH Table — single random projection table
# ──────────────────────────────────────────────
class LSHTable:
    """
    One LSH table with k-bit random Gaussian projections.
    """
    def __init__(self, table_id: int, seed: int, k: int = N_HASH_BITS):
        self.table_id = table_id
        self.k = k
        self.num_buckets = 1 << k

        # Random projection matrix: k × DIM Gaussian(0, 1), fixed seed
        rng = random.Random(seed)
        self.proj = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(k)]
        # Numpy version for fast dot product
        if _NUMPY_AVAILABLE:
            self._proj_np = np.array(self.proj, dtype=np.float64)
        else:
            self._proj_np = None

        # Buckets: bucket_id -> list of neuron_ids
        self.buckets: dict[int, list[int]] = {}
        self.lock = threading.Lock()

    def hash(self, vec: list[float]) -> int:
        """k-bit LSH: sign(v · R_i). Uses numpy if available."""
        if _NUMPY_AVAILABLE:
            v = np.array(vec, dtype=np.float64)
            dots = np.dot(self._proj_np, v)
            h = 0
            for bit in range(self.k):
                if dots[bit] >= 0:
                    h |= (1 << bit)
            return h
        # Pure Python fallback
        h = 0
        for bit in range(self.k):
            dot = 0.0
            row = self.proj[bit]
            for val, weight in zip(vec, row):
                dot += val * weight
            if dot >= 0:
                h |= (1 << bit)
        return h

    def add(self, neuron_id: int, vec: list[float]) -> None:
        bid = self.hash(vec)
        with self.lock:
            if bid not in self.buckets:
                self.buckets[bid] = []
            self.buckets[bid].append(neuron_id)

    def lookup(self, vec: list[float]) -> list[int]:
        """Return all neuron IDs in the bucket + Hamming neighbors."""
        bid = self.hash(vec)
        seen: set[int] = set()
        candidates: list[int] = []

        # Main bucket
        with self.lock:
            bucket_ids = [bid] + self._neighbors(bid)
            for b in bucket_ids:
                bucket = self.buckets.get(b)
                if bucket:
                    for nid in bucket:
                        if nid not in seen:
                            seen.add(nid)
                            candidates.append(nid)
        return candidates

    def _neighbors(self, bid: int) -> list[int]:
        """Hamming neighbors: flip each bit."""
        return [bid ^ (1 << b) for b in range(self.k)]

    def size(self) -> int:
        total = 0
        with self.lock:
            for v in self.buckets.values():
                total += len(v)
        return total

    def bucket_stats(self) -> dict:
        sizes = []
        with self.lock:
            for v in self.buckets.values():
                sizes.append(len(v))
        if not sizes:
            return {"count": 0, "avg": 0, "max": 0, "min": 0}
        return {
            "count": len(sizes),
            "avg": sum(sizes) / len(sizes),
            "max": max(sizes),
            "min": min(sizes),
        }


# ──────────────────────────────────────────────
# LSHIndex — Multi-table LSH + CellState + 2-stage selection
# ──────────────────────────────────────────────
class LSHIndex:
    """
    Multi-table LSH index with full cell state and 2-stage active selection.

    Parameters:
        L:        Number of independent LSH tables (default: 8)
        k:        Bits per table (default: 12 → 4096 buckets)
        data_dir: Directory for WAL persistence
    """

    def __init__(self, L: int = NUM_TABLES, k: int = N_HASH_BITS,
                 data_dir: str = "~/neuron-data/lsh_index"):
        self.L = L
        self.k = k
        self.data_dir = os.path.expanduser(data_dir)

        # LSH tables with different seeds
        self.tables: list[LSHTable] = []
        for t in range(L):
            seed = 1000 + t  # unique seed per table
            self.tables.append(LSHTable(table_id=t, seed=seed, k=k))

        # Full neuron storage (shared across tables)
        self.neuron_ids: list[int] = []
        self.vectors: list[tuple[float, ...]] = []
        self._id_to_index: dict[int, int] = {}

        # Cell state
        self.cells: dict[int, CellState] = {}
        self.total_neurons = 0
        self.add_count = 0

        self.lock = threading.Lock()
        self.lock_cells = threading.Lock()

        os.makedirs(self.data_dir, exist_ok=True)
        self.wal_path = os.path.join(self.data_dir, "lsh_index.wal")
        self._load_wal()

    # ── Add neuron ──────────────────────────────
    def add_neuron(self, neuron_id: int, vector: list[float],
                   context: bytes = b"", links: list[int] | None = None) -> None:
        # Add to all LSH tables
        for table in self.tables:
            table.add(neuron_id, vector)

        with self.lock:
            idx = len(self.neuron_ids)
            self.neuron_ids.append(neuron_id)
            self.vectors.append(tuple(vector))
            self._id_to_index[neuron_id] = idx
            self.total_neurons += 1
            self.add_count += 1

        # Create cell state
        with self.lock_cells:
            if neuron_id not in self.cells:
                self.cells[neuron_id] = CellState(
                    neuron_id=neuron_id, signature=vector,
                    context=context, links=links
                )

        # Write to WAL
        self._write_wal(neuron_id, vector)

    def _write_wal(self, neuron_id: int, vector: list[float]) -> None:
        record = struct.pack("!I", neuron_id)
        record += struct.pack("!H", len(vector))
        record += struct.pack(f"!{len(vector)}f", *vector)
        with open(self.wal_path, "ab") as f:
            f.write(record)
            f.flush()
            os.fsync(f.fileno())

    def _load_wal(self):
        if not os.path.exists(self.wal_path):
            return
        with open(self.wal_path, "rb") as f:
            data = f.read()
        offset = 0
        while offset + 6 <= len(data):
            neuron_id = struct.unpack("!I", data[offset: offset + 4])[0]
            vec_len = struct.unpack("!H", data[offset + 4: offset + 6])[0]
            offset += 6
            if offset + vec_len * 4 > len(data):
                break
            vector = list(struct.unpack(f"!{vec_len}f", data[offset: offset + vec_len * 4]))
            offset += vec_len * 4
            self.total_neurons += 1
            self.add_count += 1
            # Add to LSH tables
            for table in self.tables:
                table.add(neuron_id, vector)
            self.neuron_ids.append(neuron_id)
            self.vectors.append(tuple(vector))
            self._id_to_index[neuron_id] = len(self.neuron_ids) - 1

    # ── Stage 1: LSH Lookup ─────────────────────
    def lsh_lookup(self, query: list[float]) -> list[int]:
        """
        Stage 1: Union of candidates from all L tables.
        Returns deduplicated list of candidate neuron IDs.
        """
        seen: set[int] = set()
        candidates: list[int] = []

        for table in self.tables:
            for nid in table.lookup(query):
                if nid not in seen:
                    seen.add(nid)
                    candidates.append(nid)
        return candidates

    # ── Stage 2: Filter by relevance ────────────
    def filter_by_relevance(
        self,
        candidate_ids: list[int],
        query: list[float],
        threshold: float = 0.15,
        top_k: int = 10,
    ) -> list[tuple[int, float, int]]:
        """
        Stage 2: Cosine similarity + gate_bias filtering.
        1. Compute cosine similarity for each candidate
        2. Apply gate_bias: effective threshold = threshold * (1 + (gate_bias - 0.5))
        3. Return top_k by similarity above effective threshold
        """
        if not candidate_ids:
            return []

        scored: list[tuple[int, float, int]] = []

        with self.lock_cells:
            for nid in candidate_ids:
                cell = self.cells.get(nid)
                if cell is None:
                    continue
                sim = cosine_similarity(query, cell.signature)

                # Gate bias modulates the threshold
                # Neutral (0.5) → no change. Low bias (e.g. 0.2) → easier to activate.
                gate_factor = 1.0 + (cell.gate_bias - 0.5)
                effective_threshold = threshold * gate_factor

                if sim >= effective_threshold:
                    scored.append((nid, sim, 0))

        # Sort by similarity descending, return top_k
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ── 2-stage select_active_neurons ────────────
    def select_active_neurons(
        self,
        query: list[float],
        top_k: int = 10,
        threshold: float = 0.15,
    ) -> list[tuple[int, float, int]]:
        """
        Full 2-stage activation:

        Stage 1: lsh_lookup → get candidates from L multi-tables
        Stage 2: filter_by_relevance → cosine + gate_bias filtering
        """
        candidates = self.lsh_lookup(query)
        return self.filter_by_relevance(candidates, query, threshold, top_k)

    # ── Brute-force search (for benchmarking) ──
    def brute_force_search(self, query: list[float], top_k: int = 10) -> list[tuple[int, float, int]]:
        """Full scan for ground truth Precision@10 comparison."""
        scored: list[tuple[int, float, int]] = []
        with self.lock_cells:
            for nid, cell in self.cells.items():
                sim = cosine_similarity(query, cell.signature)
                scored.append((nid, sim, 0))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ── Stats ───────────────────────────────────
    def get_stats(self) -> dict:
        stats = {
            "total_neurons": self.total_neurons,
            "L": self.L,
            "k": self.k,
            "tables": [],
        }
        for t in self.tables:
            stats["tables"].append(t.bucket_stats())
        return stats


# ═══════════════════════════════════════════════
# Write functions (ported from memory_write.py)
# ═══════════════════════════════════════════════

CONTEXT_SIZE = 64
MAX_LINKS = 16
GATE_BIAS_LEARN_RATE = 0.05


def write_to_active_neurons(
    index: LSHIndex,
    active_neurons: list[tuple[int, float, int]],
    input_vector: list[float],
    context_data: bytes = b"",
    link_ids: list[int] | None = None,
) -> int:
    """Write to active neurons: moving average + context + links."""
    ctx = (context_data if isinstance(context_data, bytes) else b"")[:CONTEXT_SIZE]
    ctx_padded = (ctx + b"\x00" * CONTEXT_SIZE)[:CONTEXT_SIZE]
    now = time.time()
    updated = 0

    with index.lock_cells:
        for neuron_id, similarity, _ in active_neurons:
            cell = index.cells.get(neuron_id)
            if cell is None:
                continue

            # 1. Moving average
            ver = cell.memory_version
            alpha = 1.0 / (ver + 2.0)
            new_sig = []
            for old_val, inp_val in zip(cell.signature, input_vector[:DIM]):
                new_val = old_val * (1.0 - alpha) + inp_val * alpha
                new_sig.append(new_val)
            # Normalize
            norm = math.sqrt(sum(v * v for v in new_sig))
            if norm > 0:
                new_sig = [v / norm for v in new_sig]
            cell.signature = new_sig

            # 2. Context buffer
            cell.context = ctx_padded

            # 3. Link table (FIFO)
            if link_ids:
                existing = set(cell.links)
                new_links = [lid for lid in link_ids if lid not in existing]
                cell.links.extend(new_links)
                if len(cell.links) > MAX_LINKS:
                    cell.links = cell.links[-MAX_LINKS:]

            # 4. Increment
            cell.memory_version += 1
            cell.access_count += 1
            cell.last_accessed = now
            updated += 1

    return updated


_next_neuron_id = [100_000_000]


def create_new_neuron(
    index: LSHIndex,
    input_vector: list[float],
    context_data: bytes = b"",
    link_ids: list[int] | None = None,
) -> int:
    """Create a new neuron from input vector."""
    global _next_neuron_id
    neuron_id = _next_neuron_id[0]
    _next_neuron_id[0] += 1

    sig = list(input_vector[:DIM])
    norm = math.sqrt(sum(v * v for v in sig))
    if norm > 0:
        sig = [v / norm for v in sig]
    else:
        sig = [0.0] * DIM

    index.add_neuron(neuron_id=neuron_id, vector=sig,
                     context=context_data, links=link_ids)

    with index.lock_cells:
        if neuron_id not in index.cells:
            ctx = (context_data if isinstance(context_data, bytes) else b"")[:CONTEXT_SIZE]
            ctx_padded = (ctx + b"\x00" * CONTEXT_SIZE)[:CONTEXT_SIZE]
            cell = CellState(neuron_id, sig, ctx_padded, link_ids)
            cell.gate_bias = GATE_BIAS_DEFAULT
            index.cells[neuron_id] = cell

    return neuron_id


def update_gate_bias(cell: CellState, reward_signal: float) -> float:
    if reward_signal > 0:
        cell.gate_bias *= (1.0 - GATE_BIAS_LEARN_RATE)
    elif reward_signal < 0:
        cell.gate_bias *= (1.0 + GATE_BIAS_LEARN_RATE)
    cell.gate_bias = max(GATE_BIAS_MIN, min(GATE_BIAS_MAX, cell.gate_bias))
    cell.memory_version += 1
    return cell.gate_bias


# ═══════════════════════════════════════════════
# Benchmark: Phase 1 test — 100K cells, 100 queries
# ═══════════════════════════════════════════════
def benchmark_100k():
    """Full Phase 1 benchmark: 100K cells, 100 queries, Precision@10."""
    import shutil

    d = os.path.expanduser("~/neuron-data/lsh_bench")
    if os.path.exists(d):
        shutil.rmtree(d)

    idx = LSHIndex(L=NUM_TABLES, k=N_HASH_BITS, data_dir=d)
    rng = random.Random(42)

    N = 100_000
    N_CLUSTERS = 10

    print(f"{'=' * 60}")
    print(f"BENCHMARK: {N:,} neurons — L={idx.L}, k={idx.k}, {idx.tables[0].num_buckets:,} buckets/table")
    print(f"{'=' * 60}")

    # ── Generate clustered data ──
    print(f"\nGenerating {N:,} neurons in {N_CLUSTERS} clusters (cluster spread=0.3)...")
    per_cluster = N // N_CLUSTERS

    centroids = []
    for c in range(N_CLUSTERS):
        centroid = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v * v for v in centroid))
        if norm > 0:
            centroid = [v / norm for v in centroid]
        centroids.append(centroid)

    t0 = time.perf_counter()
    for i in range(N):
        cluster = i % N_CLUSTERS
        vec = [c + rng.gauss(0, 0.3) for c in centroids[cluster]]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        idx.add_neuron(i, vec)

        if (i + 1) % 25000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Built {i+1:,}/{N:,} ({elapsed:.1f}s)")

    build_time = time.perf_counter() - t0
    print(f"\n  Build time: {build_time:.2f}s ({N / build_time:.0f} neurons/sec)")

    # Bucket stats
    stats = idx.get_stats()
    for ti, tbl in enumerate(stats["tables"]):
        print(f"  Table {ti}: buckets={tbl['count']}, avg={tbl['avg']:.1f}, max={tbl['max']}")

    # ── 100 queries: measure latency, Precision@10 ──
    N_QUERIES = 100
    print(f"\nSearching {N_QUERIES} queries...")
    all_precisions: list[float] = []
    all_latencies: list[float] = []
    all_recall_50: list[float] = []
    all_candidate_counts: list[int] = []

    for q in range(N_QUERIES):
        # Query from a random cluster
        cid = q % N_CLUSTERS
        qv = [c + rng.gauss(0, 0.2) for c in centroids[cid]]
        norm = math.sqrt(sum(v * v for v in qv))
        if norm > 0:
            qv = [v / norm for v in qv]

        # Ground truth (brute force)
        t0 = time.perf_counter()
        brute = idx.brute_force_search(qv, top_k=50)
        brute_time = (time.perf_counter() - t0) * 1000
        brute_top10 = set(n for n, _, _ in brute[:10])
        brute_top50 = set(n for n, _, _ in brute[:50])

        # LSH 2-stage
        t0 = time.perf_counter()

        # Stage 1
        candidates = idx.lsh_lookup(qv)
        cand_time = (time.perf_counter() - t0) * 1000

        # Stage 2
        lsh_results = idx.filter_by_relevance(candidates, qv, threshold=0.0, top_k=10)
        total_time = (time.perf_counter() - t0) * 1000

        lsh_top10 = set(n for n, _, _ in lsh_results)
        lsh_top10_list = [n for n, _, _ in lsh_results]

        # Precision@10
        overlap = brute_top10 & lsh_top10
        prec = len(overlap) / 10.0 * 100

        # Recall@50: check if LSH top-10 are in brute top-50
        recall_50 = len(lsh_top10 & brute_top50) / 10.0 * 100 if len(lsh_results) > 0 else 0.0

        all_precisions.append(prec)
        all_latencies.append(total_time)
        all_recall_50.append(recall_50)
        all_candidate_counts.append(len(candidates))

    # ── Aggregate results ──
    avg_prec = sum(all_precisions) / len(all_precisions)
    avg_lat = sum(all_latencies) / len(all_latencies)
    avg_cand = sum(all_candidate_counts) / len(all_candidate_counts)
    avg_recall = sum(all_recall_50) / len(all_recall_50)

    print(f"\n{'─' * 50}")
    print(f"RESULTS: {N:,} neurons, {N_QUERIES} queries")
    print(f"{'─' * 50}")
    print(f"  Build time:          {build_time:.2f}s")
    print(f"  Avg query time:      {avg_lat:.2f}ms")
    print(f"  Avg candidates:      {avg_cand:.0f} per query")
    print(f"  Precision@10:        {avg_prec:.1f}%")
    print(f"  Recall@50:           {avg_recall:.1f}%")
    print(f"  Multi-table L:       {idx.L}")
    print(f"  Bits per table:      {idx.k}")
    print(f"  Buckets per table:   {idx.tables[0].num_buckets:,}")
    print(f"{'─' * 50}")

    # Verify > 90%
    if avg_prec >= 90.0:
        print(f"✅ PASS: Precision@10={avg_prec:.1f}% ≥ 90% — LSH works!")
    else:
        print(f"❌ FAIL: Precision@10={avg_prec:.1f}% < 90% — need tuning")
    print()

    # Save results to file
    results_path = os.path.expanduser("~/neuron-index/test_lsh_results.txt")
    with open(results_path, "w") as f:
        f.write(f"LSH Multi-Table Benchmark Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"  Neurons:     {N:,}\n")
        f.write(f"  Queries:     {N_QUERIES}\n")
        f.write(f"  L:           {idx.L}\n")
        f.write(f"  k:           {idx.k}\n")
        f.write(f"  Build time:  {build_time:.2f}s\n")
        f.write(f"  Avg query:   {avg_lat:.2f}ms\n")
        f.write(f"  Avg cand:    {avg_cand:.0f}\n")
        f.write(f"  P@10:        {avg_prec:.1f}%\n")
        f.write(f"  R@50:        {avg_recall:.1f}%\n")
        f.write(f"  Status:      {'PASS' if avg_prec >= 90.0 else 'FAIL'}\n")
    print(f"Results saved → {results_path}")

    return {
        "build_time": build_time,
        "avg_query_ms": avg_lat,
        "avg_candidates": avg_cand,
        "precision_at_10": avg_prec,
        "recall_at_50": avg_recall,
        "L": idx.L,
        "k": idx.k,
        "N": N,
    }


# ═══════════════════════════════════════════════
# 2-stage select demo
# ═══════════════════════════════════════════════
def demo_2stage_select():
    """Demonstrate 2-stage select_active_neurons on 10 random queries."""
    import shutil

    d = os.path.expanduser("~/neuron-data/lsh_select_demo")
    if os.path.exists(d):
        shutil.rmtree(d)

    idx = LSHIndex(L=NUM_TABLES, k=N_HASH_BITS, data_dir=d)
    rng = random.Random(42)

    N = 50_000
    print(f"Building {N:,} neurons...")
    for i in range(N):
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        idx.add_neuron(i, vec)

    print(f"\n{'=' * 60}")
    print(f"2-Stage Select Demo — {N:,} neurons, 10 queries")
    print(f"{'=' * 60}")

    for q in range(10):
        qv = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v * v for v in qv))
        if norm > 0:
            qv = [v / norm for v in qv]

        # Stage 1
        t0 = time.perf_counter()
        candidates = idx.lsh_lookup(qv)
        stage1_time = (time.perf_counter() - t0) * 1000

        # Stage 2
        t0 = time.perf_counter()
        active = idx.filter_by_relevance(candidates, qv, threshold=0.15, top_k=10)
        stage2_time = (time.perf_counter() - t0) * 1000

        total_time = stage1_time + stage2_time

        print(f"\n  Query {q + 1}:")
        print(f"    Stage 1 (LSH lookup):   {len(candidates):>5,} candidates ({stage1_time:.2f}ms)")
        print(f"    Stage 2 (filter+gate):  {len(active):>3} active ({stage2_time:.2f}ms)")
        print(f"    Total:                  {total_time:.2f}ms")
        if active:
            for nid, sim, _ in active[:5]:
                cell = idx.cells.get(nid)
                bias = cell.gate_bias if cell else 0
                print(f"      neuron[{nid:8d}]: sim={sim:.4f}, gate_bias={bias:.2f}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--bench" in args:
        benchmark_100k()
    elif "--select-demo" in args:
        demo_2stage_select()
    elif "--all" in args:
        print("=" * 60)
        print("PHASE 1: Full Test Suite")
        print("=" * 60)
        print()
        result = benchmark_100k()
        print()
        demo_2stage_select()
        print()
        if result["precision_at_10"] >= 90.0:
            print("✅ ALL TESTS PASS")
        else:
            print("❌ BENCHMARK FAILED — precision < 90%")
    else:
        print("new_lsh_index.py — Multi-Table LSH + Active Inference")
        print()
        print("Usage:")
        print("  python3 new_lsh_index.py --bench       100K benchmark, 100 queries")
        print("  python3 new_lsh_index.py --select-demo 2-stage select demo")
        print("  python3 new_lsh_index.py --all         Run all tests")
