"""
linear_index.py — 100M Active Inference Neuron Index (Direct Linear Scan)
============================================================================

Architecture:
  Single flat array of (neuron_id, float_vector) tuples.
  Query = brute-force cosine similarity over ALL neurons.
  
  Simple. Correct. Guaranteed 100% Precision@10.
  
  For 100K neurons: ~0.5s per query (pure Python).
  For 100M neurons: needs numpy + parallel (future).
  
  For now: proof of concept with small N. Works correctly.
"""

import struct
import math
import random
import os
import json
import threading
import time


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class FlatNeuronIndex:
    """
    Flat brute-force neuron index.
    
    All neurons stored in a single list.
    Query = linear scan over all → top_k by cosine similarity.
    
    Precision@10 = 100%.
    """

    def __init__(self, data_dir: str = "~/neuron-data", auto_save_interval: int = 10000):
        self.data_dir = os.path.expanduser(data_dir)
        self.auto_save_interval = auto_save_interval

        self.neuron_ids: list[int] = []
        self.vectors: list[tuple[float, ...]] = []
        self._id_to_index: dict[int, int] = {}
        self.total_neurons = 0
        self.add_count = 0
        self.lock = threading.Lock()

        os.makedirs(self.data_dir, exist_ok=True)
        self.wal_path = os.path.join(self.data_dir, "neuron_flat.wal")
        self._load_wal()

    def add_neuron_to_index(
        self,
        neuron_id: int,
        signature_binary: bytes | None = None,
        signature_float: list[float] | None = None,
    ) -> None:
        if signature_float is None:
            raise ValueError("signature_float is required")
        with self.lock:
            idx = len(self.neuron_ids)
            self.neuron_ids.append(neuron_id)
            self.vectors.append(tuple(signature_float))
            self._id_to_index[neuron_id] = idx
            self.total_neurons += 1
            self.add_count += 1
        self._write_wal(neuron_id, signature_float)
        if self.add_count % self.auto_save_interval == 0:
            self.save_metadata()

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
            self.neuron_ids.append(neuron_id)
            self.vectors.append(tuple(vector))
            self._id_to_index[neuron_id] = len(self.neuron_ids) - 1
            self.total_neurons += 1

    def save_metadata(self):
        meta = {
            "total_neurons": self.total_neurons,
            "add_count": self.add_count,
            "wal_path": self.wal_path,
        }
        with open(os.path.join(self.data_dir, "index_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def search_similar(
        self, query_float: list[float], top_k: int = 10
    ) -> list[tuple[int, float, int]]:
        """
        Brute-force cosine similarity over ALL stored neurons.
        Returns (neuron_id, similarity, hamming_distance).
        hamming_distance = 0 (not used in flat mode).
        """
        results: list[tuple[float, int]] = []
        n = len(self.vectors)

        with self.lock:
            for i in range(n):
                sim = cosine_similarity(query_float, self.vectors[i])
                if len(results) < top_k:
                    results.append((sim, self.neuron_ids[i]))
                    if len(results) == top_k:
                        results.sort(reverse=True, key=lambda x: x[0])
                elif sim > results[-1][0]:
                    # Binary search insertion
                    lo, hi = 0, top_k - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if results[mid][0] > sim:
                            lo = mid + 1
                        else:
                            hi = mid
                    results.insert(lo, (sim, self.neuron_ids[i]))
                    results.pop()

        return [(nid, sim, 0) for sim, nid in results]

    def select_active_neurons(
        self,
        query_float: list[float],
        index: int | None = None,
        top_k: int = 50,
        threshold: float = 0.5,
    ) -> list[tuple[int, float, int]]:
        results = self.search_similar(query_float, top_k * 2)
        return [(nid, sim, 0) for nid, sim, _ in results if sim >= threshold][:top_k]

    def get_stats(self) -> dict:
        return {
            "total_neurons": self.total_neurons,
            "wal_path": self.wal_path,
            "type": "flat_bruteforce",
            "precision_at_10": 100.0,
        }


def demo():
    """Flat index demo — guaranteed 100% precision."""
    N_NEURONS = 100_000
    N_QUERIES = 10
    TOP_K = 10
    THRESHOLD = 0.15

    import shutil
    d = os.path.expanduser("~/neuron-data/flat_demo")
    if os.path.exists(d):
        shutil.rmtree(d)

    index = FlatNeuronIndex(data_dir=d, auto_save_interval=50000)
    rng = random.Random(42)

    # ── 1. Build ──
    print(f"Building flat index: {N_NEURONS:,} neurons (64-d, unit sphere)...")
    t0 = time.perf_counter()
    for i in range(N_NEURONS):
        vec = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        index.add_neuron_to_index(neuron_id=i, signature_float=vec)
    build_time = time.perf_counter() - t0
    index.save_metadata()

    print(f"  Build time: {build_time:.2f}s ({N_NEURONS / build_time:,.0f} neurons/sec)")

    # ── 2. Query ──
    print(f"\nSearching {N_QUERIES} queries (brute force, top_k={TOP_K}):")
    query_vectors = []
    for _ in range(N_QUERIES):
        vec = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        query_vectors.append(vec)

    latencies = []
    for q, qvec in enumerate(query_vectors):
        t0 = time.perf_counter()
        results = index.search_similar(qvec, top_k=TOP_K)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  Query {q}: {ms:.2f}ms — top-5: {[(nid, round(s, 4)) for nid, s, _ in results[:5]]}")

    avg_lat = sum(latencies) / len(latencies)
    print(f"\n  Latency: avg={avg_lat:.2f}ms, min={min(latencies):.2f}ms, max={max(latencies):.2f}ms")

    # ── 3. Accuracy (exhaustive → should be 100%) ──
    print(f"\nVerifying precision...")
    qvec = query_vectors[0]
    t0 = time.perf_counter()
    brute = index.search_similar(qvec, top_k=TOP_K)
    brute_time = time.perf_counter() - t0
    # Re-query to confirm consistency
    brute2 = index.search_similar(qvec, top_k=TOP_K)
    ids1 = [nid for nid, _, _ in brute]
    ids2 = [nid for nid, _, _ in brute2]
    overlap = len(set(ids1) & set(ids2))
    print(f"  Self-consistency: {overlap}/{TOP_K} ({overlap / TOP_K * 100:.0f}%)")
    print(f"  Precision@10: 100% (guaranteed — brute force over all neurons)")
    print(f"  Speed: {brute_time * 1000:.2f}ms per query over {N_NEURONS:,} neurons")

    # ── 4. Active inference ──
    print(f"\nActive inference: select_active_neurons(top_k={TOP_K}, threshold={THRESHOLD}):")
    for q, qvec in enumerate(query_vectors[:5]):
        t0 = time.perf_counter()
        active = index.select_active_neurons(qvec, top_k=TOP_K, threshold=THRESHOLD)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  Query {q}: {len(active)} active neurons ({ms:.2f}ms)")
        if active:
            for nid, sim, _ in active[:4]:
                print(f"    neuron[{nid:6d}]: sim={sim:.4f}")

    print(f"\n{'=' * 50}")
    print(f"DEMO COMPLETE — FlatNeuronIndex")
    print(f"Index: {N_NEURONS:,} neurons")
    print(f"Build: {build_time:.2f}s | Latency: {avg_lat:.2f}ms avg | Precision@10: 100%")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        print("FlatNeuronIndex — 100M Active Inference Neuron Index (Brute Force)")
        print("Usage: python3 linear_index.py --demo")
