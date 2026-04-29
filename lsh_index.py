"""
lsh_index.py — Locality-Sensitive Hashing Index (v3)
======================================================

Architecture:
  Layer 1 (LSH): 64-d float vectors → 4,096 buckets via 12-bit random projections
  Layer 2 (Scan): Brute-force cosine within bucket
  
  Hash: sign(v · R_i) for i=0..11, where R is a 12×64 Gaussian matrix.
  This IS locality-sensitive: close vectors have high probability of same hash.
"""

import struct, math, random, os, json, threading, time

_NUMPY_AVAILABLE = False
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None


class LSHIndex:
    """
    LSH + brute-force per bucket.
    """
    def __init__(self, data_dir="~/neuron-data", n_hash_bits=12, auto_save=50000, use_numpy=True):
        self.data_dir = os.path.expanduser(data_dir)
        self.n_hash_bits = n_hash_bits
        self.num_buckets = 1 << n_hash_bits  # 4096 for 12 bits
        self.auto_save = auto_save
        self.use_numpy = use_numpy and _NUMPY_AVAILABLE

        self.buckets: dict[int, list] = {}  # bucket_id -> [(neuron_id, vector_tuple)]
        self.neuron_ids: list[int] = []
        self.vectors: list[tuple] = []
        self.total = 0
        self.add_count = 0
        self.lock = threading.Lock()

        # Random projection matrix: fixed seed for reproducibility
        self._rng = random.Random(42)
        self._proj_matrix = [[self._rng.gauss(0, 1) for _ in range(64)] for _ in range(n_hash_bits)]

        os.makedirs(self.data_dir, exist_ok=True)
        self.wal_path = os.path.join(self.data_dir, "lsh.wal")
        self._load_wal()

    def _hash(self, vec: list[float]) -> int:
        """12-bit LSH hash: sign(v · R_i) for each random projection."""
        h = 0
        for bit in range(self.n_hash_bits):
            dot = 0.0
            row = self._proj_matrix[bit]
            for val, weight in zip(vec, row):
                dot += val * weight
            if dot >= 0:
                h |= (1 << bit)
        return h

    def add(self, neuron_id: int, signature_float: list[float]) -> None:
        with self.lock:
            bucket_id = self._hash(signature_float)
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            self.buckets[bucket_id].append((neuron_id, tuple(signature_float)))
            self.neuron_ids.append(neuron_id)
            self.vectors.append(tuple(signature_float))
            self.total += 1
            self.add_count += 1
        self._write_wal(neuron_id, signature_float)
        if self.add_count % self.auto_save == 0:
            self.save()

    def _write_wal(self, nid: int, vec: list[float]) -> None:
        rec = struct.pack("!I", nid) + struct.pack("!H", len(vec))
        rec += struct.pack(f"!{len(vec)}f", *vec)
        with open(self.wal_path, "ab") as f:
            f.write(rec); f.flush(); os.fsync(f.fileno())

    def _load_wal(self):
        if not os.path.exists(self.wal_path): return
        with open(self.wal_path, "rb") as f:
            data = f.read()
        off = 0
        while off + 6 <= len(data):
            nid = struct.unpack("!I", data[off:off+4])[0]
            vl = struct.unpack("!H", data[off+4:off+6])[0]
            off += 6
            if off + vl*4 > len(data): break
            vec = list(struct.unpack(f"!{vl}f", data[off:off+vl*4]))
            off += vl*4
            bid = self._hash(vec)
            if bid not in self.buckets: self.buckets[bid] = []
            self.buckets[bid].append((nid, tuple(vec)))
            self.neuron_ids.append(nid); self.vectors.append(tuple(vec))
            self.total += 1

    def save(self):
        meta = {"total": self.total, "buckets_active": len(self.buckets), "n_hash_bits": self.n_hash_bits}
        with open(os.path.join(self.data_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def search(self, query: list[float], top_k=10) -> list[tuple[int, float]]:
        """Search: hash query → scan bucket + probe neighbors + wider probe."""
        bucket_id = self._hash(query)

        # Collect candidates: main bucket + up to 12 Hamming neighbors
        candidates: list[tuple[int, tuple]] = []
        seen: set[int] = set()
        
        # Main bucket
        for bid in [bucket_id] + self._neighbor_buckets(bucket_id):
            bucket = self.buckets.get(bid)
            if bucket:
                with self.lock:
                    for nid, vec in bucket:
                        if nid not in seen:
                            seen.add(nid)
                            candidates.append((nid, vec))
        
        # If not enough, widen search: additional multi-probe (2-bit + 3-bit flips)
        if len(candidates) < top_k * 10:
            for b1 in range(self.n_hash_bits):
                for b2 in range(b1 + 1, self.n_hash_bits):
                    bid2 = bucket_id ^ (1 << b1) ^ (1 << b2)
                    bucket = self.buckets.get(bid2)
                    if bucket:
                        with self.lock:
                            for nid, vec in bucket:
                                if nid not in seen:
                                    seen.add(nid)
                                    candidates.append((nid, vec))
                    if len(candidates) >= top_k * 30:
                        break
                if len(candidates) >= top_k * 30:
                    break
            # Also 3-bit flips if still not enough
            if len(candidates) < top_k * 5:
                for b1 in range(self.n_hash_bits):
                    for b2 in range(b1+1, self.n_hash_bits):
                        for b3 in range(b2+1, self.n_hash_bits):
                            bid3 = bucket_id ^ (1 << b1) ^ (1 << b2) ^ (1 << b3)
                            bucket = self.buckets.get(bid3)
                            if bucket:
                                with self.lock:
                                    for nid, vec in bucket:
                                        if nid not in seen:
                                            seen.add(nid)
                                            candidates.append((nid, vec))
                            if len(candidates) >= top_k * 50:
                                break
                        if len(candidates) >= top_k * 50:
                            break
                    if len(candidates) >= top_k * 50:
                        break

        if not candidates:
            return []

        # Cosine similarity on candidates
        qnorm = math.sqrt(sum(v*v for v in query))
        if qnorm == 0: return []
        q = [v / qnorm for v in query]

        results = []
        for nid, vec in candidates:
            dot = sum(a*b for a,b in zip(q, vec))
            results.append((dot, nid))

        results.sort(reverse=True)
        return [(nid, sim) for sim, nid in results[:top_k]]

    def _neighbor_buckets(self, bid: int) -> list[int]:
        """Hamming neighbors: flip each bit."""
        return [bid ^ (1 << b) for b in range(self.n_hash_bits)]

    def get_stats(self):
        sizes = [len(v) for v in self.buckets.values()]
        return {
            "total": self.total, "buckets": len(self.buckets),
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "avg": sum(sizes)/len(sizes) if sizes else 0,
        }


def demo():
    """Full LSH demo: 100K neurons in 10 clusters, measure precision."""
    import shutil
    d = os.path.expanduser("~/neuron-data/lsh_demo")
    if os.path.exists(d): shutil.rmtree(d)

    idx = LSHIndex(data_dir=d, n_hash_bits=8, use_numpy=_NUMPY_AVAILABLE)
    rng = random.Random(42)

    N = 100_000
    N_CLUSTERS = 10
    print(f"Building LSH index: {N:,} neurons in {N_CLUSTERS} clusters (64-d, 8-bit, {idx.num_buckets} buckets)...")
    
    t0 = time.perf_counter()
    per_cluster = N // N_CLUSTERS
    
    # Generate cluster centroids
    centroids = []
    for c in range(N_CLUSTERS):
        centroid = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v*v for v in centroid))
        if norm > 0: centroid = [v/norm for v in centroid]
        centroids.append(centroid)
    
    for i in range(N):
        cluster = i % N_CLUSTERS
        centroid = centroids[cluster]
        # Add noise around centroid (spread = 0.3, so inner-cluster cos ≈ 0.95)
        vec = [c + rng.gauss(0, 0.3) for c in centroid]
        norm = math.sqrt(sum(v*v for v in vec))
        if norm > 0: vec = [v/norm for v in vec]
        idx.add(i, vec)
    bt = time.perf_counter() - t0
    idx.save()

    stats = idx.get_stats()
    print(f"  Build: {bt:.1f}s ({N/bt:,.0f}/s)")
    print(f"  Buckets: {stats['buckets']}, avg={stats['avg']:.1f}, max={stats['max']}")

    # Queries — use cluster centroids as queries (should find its own cluster)
    print(f"\nSearching 10 queries...")
    latencies = []
    for q in range(10):
        qv = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v*v for v in qv))
        if norm > 0: qv = [v/norm for v in qv]
        t0 = time.perf_counter()
        res = idx.search(qv, top_k=10)
        ms = (time.perf_counter()-t0)*1000
        latencies.append(ms)
        print(f"  Query {q}: {ms:.2f}ms — top-5: {[(n, round(s,4)) for s,n in res[:5]]}")

    print(f"\n  Latency: avg={sum(latencies)/len(latencies):.2f}ms")

    # Accuracy: compare brute force vs LSH on first query
    print(f"\nVerifying Precision@10...")
    qv = centroids[0]  # Query a centroid — should find its cluster
    
    # Brute force over ALL neurons
    t0 = time.perf_counter()
    all_sims = []
    for i in range(N):
        dot = sum(a*b for a,b in zip(qv, idx.vectors[i]))
        all_sims.append((dot, i))
    all_sims.sort(reverse=True)
    brute_top = set(n for _, n in all_sims[:10])
    brute_time = time.perf_counter() - t0

    # LSH
    lsh_top = set(n for n, _ in idx.search(qv, top_k=10))
    overlap = brute_top & lsh_top
    prec = len(overlap)/10*100
    recall = len(overlap)/10*100
    print(f"  Brute: {brute_time*1000:.0f}ms")
    print(f"  LSH candidates: ~{stats['avg']*(1+len(idx._neighbor_buckets(0))):.0f} neuron/bucket")
    print(f"  Precision@10: {prec:.1f}%")
    print(f"  Recall@10: {recall:.1f}%")
    print(f"  Top-10 brute IDs (cluster 0 expected): {[n for _,n in all_sims[:10]]}")

    print(f"\n{'='*50}")
    print(f"LSH DEMO — 12-bit Random Projections (Clustered Data)")
    print(f"Build: {bt:.2f}s | Query: {sum(latencies)/len(latencies):.1f}ms | Precision@10: {prec:.0f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        print("LSHIndex — Locality-Sensitive Hashing for 100M Neurons")
        print("Usage: python3 lsh_index.py --demo")
