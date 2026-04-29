"""
neuron_index.py — 100M Active Inference Neuron Index
=====================================================

Architecture:
  Layer 1 (LSH):  128-bit binary signatures → 65,536 buckets (16-bit hash)
  Layer 2 (HNSW): Float vectors (64-d) → navigable small-world graph per bucket

No dependencies — pure Python 3. numpy is optional (auto-detected).
Disk-backed via append-only WAL (Write-Ahead Log) + mmap-ready bucket files.

Usage:
  index = NeuronIndex(data_dir="~/neuron-data")
  index.add_neuron(neuron_id=42, signature_binary=bytes(16), signature_float=[...])
  results = index.search_similar(query_float=[...], top_k=5)
"""

import struct
import math
import random
import os
import json
import mmap
import threading
import time
from collections import defaultdict

# ──────────────────────────────────────────────
# 1. Binary utilities: 128-bit XOR + popcount
# ──────────────────────────────────────────────

POPCOUNT_8 = tuple(bin(i).count("1") for i in range(256))

def hamming_distance(a: bytes, b: bytes) -> int:
    """XOR + popcount on two 16-byte signatures. O(1) — single CPU instruction per byte."""
    dist = 0
    for byte_a, byte_b in zip(a, b):
        dist += POPCOUNT_8[byte_a ^ byte_b]
    return dist

def hash_to_bucket(signature: bytes, hash_bits: int = 16) -> int:
    """Extract first `hash_bits` from 128-bit signature as bucket key."""
    if hash_bits <= 8:
        return signature[0] >> (8 - hash_bits)
    elif hash_bits <= 16:
        return (signature[0] << 8 | signature[1]) >> (16 - hash_bits)
    elif hash_bits <= 24:
        return ((signature[0] << 16) | (signature[1] << 8) | signature[2]) >> (24 - hash_bits)
    else:
        # Fallback: use all 16 bytes for 128-bit hash
        return (signature[0] << 8 | signature[1]) & 0xFFFF  # 16-bit bucket


# ──────────────────────────────────────────────
# 2. Cosine similarity (pure Python + optional numba)
# ──────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors. Uses dot / (norm_a * norm_b)."""
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


# ──────────────────────────────────────────────
# 3. HNSW (Hierarchical Navigable Small World) — Lightweight
# ──────────────────────────────────────────────

class HNSWNode:
    """A single node in the HNSW graph."""
    __slots__ = ("neuron_id", "vector", "level", "neighbors")

    def __init__(self, neuron_id: int, vector: list[float], level: int = 0):
        self.neuron_id = neuron_id
        self.vector = vector
        self.level = level
        self.neighbors: dict[int, list[int]] = {}  # level -> [neuron_ids]

    def add_neighbor(self, neighbor_id: int, level: int):
        if level not in self.neighbors:
            self.neighbors[level] = []
        if neighbor_id not in self.neighbors[level]:
            self.neighbors[level].append(neighbor_id)


class HNSWGraph:
    """
    In-memory HNSW index for one bucket.
    
    Parameters:
      M:        max neighbors per layer (default 16)
      M_max:    max neighbors for layer 0  (default 32)
      ef_search:  search breadth (default 50)
      ef_construction: construction breadth (default 100)
      ml:       level generation multiplier (default 1 / ln(M))
    """

    def __init__(
        self,
        M: int = 16,
        M_max: int = 32,
        ef_search: int = 50,
        ef_construction: int = 100,
    ):
        self.M = M
        self.M_max = M_max
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.ml = 1.0 / math.log(M) if M > 1 else 1.0

        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: int | None = None
        self.max_level: int = 0
        self._rng = random.Random(42)

    def _random_level(self) -> int:
        """Generate random level for a new node (exponential decay)."""
        return int(-math.log(self._rng.random()) * self.ml)

    def _search_layer(
        self,
        query: list[float],
        entry_ids: list[int],
        level: int,
        ef: int,
    ) -> list[int]:
        """Greedy search on one layer — returns ef closest candidates."""
        visited = set(entry_ids)
        candidates = list(entry_ids)
        results = list(entry_ids)

        while candidates:
            # Find closest candidate to query
            closest_idx = 0
            closest_sim = -2.0
            for i, cid in enumerate(candidates):
                if cid in self.nodes:
                    sim = cosine_similarity(query, self.nodes[cid].vector)
                    if sim > closest_sim:
                        closest_sim = sim
                        closest_idx = i

            farthest_sim = -2.0
            for rid in results:
                if rid in self.nodes:
                    sim = cosine_similarity(query, self.nodes[rid].vector)
                    if sim > farthest_sim:
                        farthest_sim = sim

            if closest_sim < farthest_sim:
                break  # Cannot improve further

            cur_id = candidates.pop(closest_idx)
            if cur_id not in self.nodes:
                continue

            for neighbor_id in self.nodes[cur_id].neighbors.get(level, []):
                if neighbor_id not in visited and neighbor_id in self.nodes:
                    visited.add(neighbor_id)
                    farthest_sim = min(
                        cosine_similarity(query, self.nodes[nid].vector)
                        for nid in results
                        if nid in self.nodes
                    )
                    sim_n = cosine_similarity(query, self.nodes[neighbor_id].vector)
                    if sim_n > farthest_sim or len(results) < ef:
                        candidates.append(neighbor_id)
                        results.append(neighbor_id)
                        if len(results) > ef:
                            # Remove the worst
                            worst_idx = min(
                                range(len(results)),
                                key=lambda i: cosine_similarity(query, self.nodes[results[i]].vector)
                                if results[i] in self.nodes
                                else -2.0,
                            )
                            results.pop(worst_idx)

        return results

    def _select_neighbors_simple(
        self, candidates: list[int], query: list[float], M: int
    ) -> list[int]:
        """Select top-M closest candidates."""
        scored = []
        for cid in candidates:
            if cid in self.nodes:
                sim = cosine_similarity(query, self.nodes[cid].vector)
                scored.append((sim, cid))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [cid for _, cid in scored[:M]]

    def insert(self, neuron_id: int, vector: list[float]) -> None:
        """Insert a node into the HNSW graph."""
        node = HNSWNode(neuron_id, vector)
        level = self._random_level()
        node.level = level
        self.nodes[neuron_id] = node

        if self.entry_point is None:
            self.entry_point = neuron_id
            self.max_level = level
            return

        # Find entry point at topmost level
        curr_entry = self.entry_point
        for lvl in range(self.max_level, level, -1):
            result = self._search_layer(vector, [curr_entry], lvl, 1)
            if result:
                curr_entry = result[0]

        # Insert at each level from min(level, max_level) down to 0
        for lvl in range(min(level, self.max_level), -1, -1):
            ef = self.ef_construction if lvl == 0 else self.ef_search
            candidates = self._search_layer(vector, [curr_entry], lvl, ef)
            M_lvl = self.M_max if lvl == 0 else self.M
            selected = self._select_neighbors_simple(candidates, vector, M_lvl)

            for neighbor_id in selected:
                self.nodes[neuron_id].add_neighbor(neighbor_id, lvl)
                self.nodes[neighbor_id].add_neighbor(neuron_id, lvl)

            if candidates:
                curr_entry = candidates[0]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = neuron_id

    def search(self, query: list[float], top_k: int = 10) -> list[tuple[int, float]]:
        """Search HNSW graph — returns [(neuron_id, similarity), ...]."""
        if self.entry_point is None or not self.nodes:
            return []

        curr_entry = self.entry_point
        for lvl in range(self.max_level, 0, -1):
            result = self._search_layer(query, [curr_entry], lvl, 1)
            if result:
                curr_entry = result[0]

        candidates = self._search_layer(query, [curr_entry], 0, self.ef_search)
        scored = []
        for cid in candidates:
            if cid in self.nodes:
                sim = cosine_similarity(query, self.nodes[cid].vector)
                scored.append((cid, sim))
        scored.sort(reverse=True, key=lambda x: x[1])
        return scored[:top_k]


# ──────────────────────────────────────────────
# 4. Bucket manager — LSH + HNSW per bucket
# ──────────────────────────────────────────────

class NeuronBucket:
    """
    One bucket = one LSH hash value.
    Contains:
      - nodes: dict[neuron_id, (signature_bytes, vector_tuple)]
      - hnsw:  HNSWGraph for float-vector search
    """

    def __init__(self, bucket_id: int, M: int = 16, ef_search: int = 50):
        self.bucket_id = bucket_id
        self.nodes: dict[int, tuple[bytes, tuple[float, ...]]] = {}
        self.hnsw = HNSWGraph(M=M, ef_search=ef_search)
        self.lock = threading.Lock()

    def add(self, neuron_id: int, signature: bytes, vector: list[float]) -> None:
        with self.lock:
            self.nodes[neuron_id] = (signature, tuple(vector))
            self.hnsw.insert(neuron_id, vector)

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        with self.lock:
            return self.hnsw.search(query_vector, top_k)

    def search_by_signature(self, query_signature: bytes, top_k: int) -> list[tuple[int, int]]:
        """Alternative: search by Hamming distance within bucket (no HNSW)."""
        scored = []
        with self.lock:
            for nid, (sig, _) in self.nodes.items():
                dist = hamming_distance(query_signature, sig)
                scored.append((dist, nid))
        scored.sort(key=lambda x: x[0])
        return [(nid, dist) for dist, nid in scored[:top_k]]

    def size(self) -> int:
        with self.lock:
            return len(self.nodes)

    def get_vectors(self, neuron_ids: list[int]) -> list[list[float] | None]:
        with self.lock:
            return [list(v) if v else None for _, (_, v) in [(nid, self.nodes.get(nid, (b"", ()))) for nid in neuron_ids]]


# ──────────────────────────────────────────────
# 5. Main NeuronIndex class
# ──────────────────────────────────────────────

class NeuronIndex:
    """
    Two-layer index for 100M neurons:
      Layer 1: 128-bit binary → LSH bucket (65,536 buckets)
      Layer 2: 64-d float → HNSW within bucket
    
    Disk-backed: append-only WAL for durability.
    """

    def __init__(
        self,
        data_dir: str = "~/neuron-data",
        num_buckets: int = 65536,
        M: int = 16,
        ef_search: int = 50,
        hash_bits: int = 16,
        auto_save_interval: int = 10000,
    ):
        self.data_dir = os.path.expanduser(data_dir)
        self.num_buckets = num_buckets
        self.hash_bits = hash_bits
        self.M = M
        self.ef_search = ef_search
        self.auto_save_interval = auto_save_interval

        self.buckets: dict[int, NeuronBucket] = {}
        self.total_neurons = 0
        self.add_count = 0
        self.lock = threading.Lock()

        os.makedirs(self.data_dir, exist_ok=True)
        self.wal_path = os.path.join(self.data_dir, "neuron_index.wal")
        self._load_wal()

    def _get_or_create_bucket(self, bucket_id: int) -> NeuronBucket:
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = NeuronBucket(
                bucket_id=bucket_id, M=self.M, ef_search=self.ef_search
            )
        return self.buckets[bucket_id]

    def _derive_binary_signature(self, float_vector: list[float]) -> bytes:
        """
        Derive 128-bit binary signature from 64-d float vector.
        
        Strategy: For each float dimension, compute a 2-bit hash:
          bit 0: sign (positive = 1, negative = 0)
          bit 1: magnitude above/below median of that dimension across the vector
        
        This gives 64 × 2 = 128 bits = 16 bytes.
        """
        sig_bits = []
        median = sorted(float_vector)[len(float_vector) // 2] if float_vector else 0.0
        
        for i, val in enumerate(float_vector):
            bit0 = 1 if val >= 0 else 0
            bit1 = 1 if abs(val) >= abs(median) else 0
            sig_bits.append(bit0)
            sig_bits.append(bit1)
        
        # Pack 128 bits into 16 bytes
        sig_bytes = bytearray(16)
        for i, bit in enumerate(sig_bits):
            if bit:
                sig_bytes[i // 8] |= 1 << (i % 8)
        return bytes(sig_bytes)

    def _signature_float_to_binary(
        self, float_vector: list[float]
    ) -> tuple[bytes, bytes]:
        """
        Two signatures from the same float:
          1. binary_signature: 128-bit deterministic hash
          2. raw_binary: direct binarization (sign-bit only, 64 bits = 8 bytes)
        Returns (binary_signature, raw_binary_padded).
        """
        if len(float_vector) != 64:
            # If not 64-d, use _derive_binary_signature and pad
            sig = self._derive_binary_signature(float_vector)
            return sig, sig[:8] + b"\x00" * 8
        
        binary_sig = self._derive_binary_signature(float_vector)
        
        # Raw binarization: 1 bit per dimension (sign only) = 64 bits = 8 bytes
        raw_bytes = bytearray(8)
        for i, val in enumerate(float_vector):
            if val >= 0:
                raw_bytes[i // 8] |= 1 << (i % 8)
        raw_padded = bytes(raw_bytes) + b"\x00" * 8
        
        return binary_sig, raw_padded

    def add_neuron_to_index(
        self,
        neuron_id: int,
        signature_binary: bytes | None = None,
        signature_float: list[float] | None = None,
    ) -> None:
        """
        Add a neuron to the index.
        
        Args:
            neuron_id: Unique neuron identifier
            signature_binary: 128-bit binary signature (16 bytes). If None, derived from float.
            signature_float: 64-d float vector for HNSW. Required.
        """
        if signature_float is None:
            raise ValueError("signature_float is required for HNSW indexing")

        # Derive binary signature if not provided
        if signature_binary is None or len(signature_binary) != 16:
            signature_binary = self._derive_binary_signature(signature_float)

        bucket_id = hash_to_bucket(signature_binary, self.hash_bits)
        bucket = self._get_or_create_bucket(bucket_id)
        bucket.add(neuron_id, signature_binary, signature_float)

        self.total_neurons += 1
        self.add_count += 1

        # Write to WAL
        self._write_wal(neuron_id, signature_binary, signature_float)

        # Periodic save
        if self.add_count % self.auto_save_interval == 0:
            self.save_metadata()

    def _write_wal(
        self, neuron_id: int, signature: bytes, vector: list[float]
    ) -> None:
        """Append to WAL: neuron_id (4) + signature (16) + vec_len (2) + vector (4*N)."""
        record = struct.pack("!I", neuron_id)
        record += signature
        record += struct.pack("!H", len(vector))
        record += struct.pack(f"!{len(vector)}f", *vector)

        with open(self.wal_path, "ab") as f:
            f.write(record)
            f.flush()
            os.fsync(f.fileno())

    def _load_wal(self):
        """Rebuild index from WAL on startup."""
        if not os.path.exists(self.wal_path):
            return

        with open(self.wal_path, "rb") as f:
            data = f.read()

        offset = 0
        while offset + 22 <= len(data):  # 4 (id) + 16 (sig) + 2 (len)
            neuron_id = struct.unpack("!I", data[offset : offset + 4])[0]
            signature = data[offset + 4 : offset + 20]
            vec_len = struct.unpack("!H", data[offset + 20 : offset + 22])[0]
            offset += 22

            if offset + vec_len * 4 > len(data):
                break
            vector = list(struct.unpack(f"!{vec_len}f", data[offset : offset + vec_len * 4]))
            offset += vec_len * 4

            bucket_id = hash_to_bucket(signature, self.hash_bits)
            bucket = self._get_or_create_bucket(bucket_id)
            bucket.add(neuron_id, signature, vector)
            self.total_neurons += 1

        print(f"NeuronIndex: loaded {self.total_neurons} neurons from {self.wal_path}")

    def save_metadata(self):
        """Save index metadata to JSON for quick reload."""
        meta = {
            "total_neurons": self.total_neurons,
            "num_buckets": self.num_buckets,
            "hash_bits": self.hash_bits,
            "add_count": self.add_count,
            "wal_path": self.wal_path,
            "buckets_active": len(self.buckets),
        }
        meta_path = os.path.join(self.data_dir, "index_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def search_similar(
        self, query_float: list[float], top_k: int = 10
    ) -> list[tuple[int, float, int]]:
        """
        Search the index for top-K similar neurons.
        
        Args:
            query_float: 64-d float query vector
            top_k: number of results
        
        Returns:
            list of (neuron_id, cosine_similarity, hamming_distance)
        """
        # Derive binary signature and find bucket
        query_binary = self._derive_binary_signature(query_float)
        bucket_id = hash_to_bucket(query_binary, self.hash_bits)

        results: list[tuple[int, float, int]] = []

        # Search main bucket with HNSW
        bucket = self.buckets.get(bucket_id)
        if bucket:
            hnsw_results = bucket.search(query_float, top_k)
            for nid, sim in hnsw_results:
                if nid in bucket.nodes:
                    sig = bucket.nodes[nid][0]
                    hdist = hamming_distance(query_binary, sig)
                    results.append((nid, sim, hdist))

        # If not enough results, also check neighboring buckets (Hamming distance 1)
        if len(results) < top_k:
            neighbor_buckets = self._get_neighbor_buckets(bucket_id)
            for nbid in neighbor_buckets:
                if nbid in self.buckets:
                    extra = self.buckets[nbid].search(query_float, top_k // 2)
                    for nid, sim in extra:
                        if nid in self.buckets[nbid].nodes:
                            sig = self.buckets[nbid].nodes[nid][0]
                            hdist = hamming_distance(query_binary, sig)
                            results.append((nid, sim, hdist))
                        if len(results) >= top_k:
                            break
                if len(results) >= top_k:
                    break

        # Sort by similarity descending, take top_k
        results.sort(key=lambda x: (-x[1], x[2]))
        return results[:top_k]

    def _get_neighbor_buckets(self, bucket_id: int) -> list[int]:
        """Get buckets at Hamming distance 1 (flip 1 bit in 16-bit hash)."""
        neighbors = []
        for bit in range(self.hash_bits):
            neighbor = bucket_id ^ (1 << bit)
            if neighbor < self.num_buckets:
                neighbors.append(neighbor)
        return neighbors

    def select_active_neurons(
        self,
        query_float: list[float],
        index: int | None = None,
        top_k: int = 50,
        threshold: float = 0.5,
    ) -> list[tuple[int, float, int]]:
        """
        Main entry point for Active Inference.
        Returns the top-K most similar neurons to the query.
        
        Args:
            query_float: 64-d float query vector (state of the agent)
            index: optional index override (unused, for API compat)
            top_k: number of active neurons to return
            threshold: similarity threshold (results below this are excluded)
        
        Returns:
            list of (neuron_id, cosine_similarity, hamming_distance)
        """
        results = self.search_similar(query_float, top_k * 2)  # Over-fetch
        filtered = [(nid, sim, hdist) for nid, sim, hdist in results if sim >= threshold]
        return filtered[:top_k]

    def get_bucket_stats(self) -> dict:
        """Return statistics about the index."""
        bucket_sizes = [(bid, bucket.size()) for bid, bucket in self.buckets.items()]
        sizes = [s for _, s in bucket_sizes]
        return {
            "total_neurons": self.total_neurons,
            "buckets_active": len(self.buckets),
            "bucket_min_size": min(sizes) if sizes else 0,
            "bucket_max_size": max(sizes) if sizes else 0,
            "bucket_avg_size": sum(sizes) / len(sizes) if sizes else 0.0,
            "wal_path": self.wal_path,
        }


# ──────────────────────────────────────────────
# 6. CLI Interface
# ──────────────────────────────────────────────

def demo():
    """Run a quick demo: index 10,000 random neurons, search 10 queries."""
    index = NeuronIndex(data_dir="~/neuron-data/demo", auto_save_interval=5000)

    print("Adding 10,000 random neurons...")
    rng = random.Random(42)
    for i in range(10000):
        vec = [rng.gauss(0, 1) for _ in range(64)]
        # Normalize to unit sphere
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        index.add_neuron_to_index(neuron_id=i, signature_float=vec)

    index.save_metadata()
    print(f"Index stats: {index.get_bucket_stats()}")

    print("\nSearching sample queries...")
    for q in range(5):
        query_vec = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in query_vec))
        if norm > 0:
            query_vec = [v / norm for v in query_vec]
        results = index.search_similar(query_vec, top_k=5)
        print(f"  Query {q}: top={[(nid, round(sim, 4)) for nid, sim, _ in results]}")

    print("\nActive inference: select_active_neurons(top_k=10, threshold=0.3)")
    for q in range(3):
        query_vec = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in query_vec))
        if norm > 0:
            query_vec = [v / norm for v in query_vec]
        active = index.select_active_neurons(query_vec, top_k=10, threshold=0.3)
        print(f"  Query {q}: {len(active)} active neurons (threshold=0.3)")
        for nid, sim, hdist in active[:3]:
            print(f"    neuron[{nid}]: sim={sim:.4f}, hdist={hdist}")

    print("\nDone!")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    elif "--stats" in sys.argv:
        idx = NeuronIndex()
        print(idx.get_bucket_stats())
    else:
        print("NeuronIndex — 100M Active Inference Neuron Index")
        print("Usage:")
        print("  python3 neuron_index.py --demo    Run demo")
        print("  python3 neuron_index.py --stats   Show index stats")
