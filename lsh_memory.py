"""
lsh_memory.py — LSHIndex + Online Learning Write Mechanism
============================================================

Combines LSH index (lsh_index.py) with online learning (memory_write.py patterns).
"""

import struct, math, random, os, json, threading, time
from typing import TYPE_CHECKING

_NUMPY_AVAILABLE = False
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None


# ═══════════════════════════════════════════════
# LSH Index (from lsh_index.py, same code)
# ═══════════════════════════════════════════════

class LSHIndex:
    def __init__(self, data_dir="~/neuron-data", n_hash_bits=8, auto_save=50000, use_numpy=True):
        self.data_dir = os.path.expanduser(data_dir)
        self.n_hash_bits = n_hash_bits
        self.num_buckets = 1 << n_hash_bits
        self.auto_save = auto_save
        self.use_numpy = use_numpy and _NUMPY_AVAILABLE
        self.buckets: dict[int, list] = {}
        self.neuron_ids: list[int] = []
        self.vectors: list[tuple] = []
        self.cells: dict[int, dict] = {}  # neuron_id -> state
        self.total = 0
        self.add_count = 0
        self.lock = threading.Lock()
        self._rng = random.Random(42)
        self._proj_matrix = [[self._rng.gauss(0,1) for _ in range(64)] for _ in range(n_hash_bits)]
        os.makedirs(self.data_dir, exist_ok=True)
        self.wal_path = os.path.join(self.data_dir, "lsh_memory.wal")
        self._load_wal()

    def _hash(self, vec):
        h = 0
        for bit in range(self.n_hash_bits):
            dot = sum(v * w for v, w in zip(vec, self._proj_matrix[bit]))
            if dot >= 0: h |= (1 << bit)
        return h

    def add(self, neuron_id, signature_float, context=b"", links=None):
        with self.lock:
            bid = self._hash(signature_float)
            if bid not in self.buckets: self.buckets[bid] = []
            self.buckets[bid].append((neuron_id, tuple(signature_float)))
            self.neuron_ids.append(neuron_id)
            self.vectors.append(tuple(signature_float))
            if neuron_id not in self.cells:
                ctx = (context if isinstance(context, bytes) else b"")[:64]
                self.cells[neuron_id] = {
                    "signature": list(signature_float),
                    "context": (ctx + b"\x00"*64)[:64],
                    "links": (links or [])[:16],
                    "gate_bias": 0.5,
                    "access_count": 0,
                    "memory_version": 0,
                    "last_accessed": 0.0,
                }
            self.total += 1
            self.add_count += 1
        self._write_wal(neuron_id, signature_float)
        if self.add_count % self.auto_save == 0: self.save()

    def _write_wal(self, nid, vec):
        rec = struct.pack("!I", nid) + struct.pack("!H", len(vec))
        rec += struct.pack(f"!{len(vec)}f", *vec)
        with open(self.wal_path, "ab") as f:
            f.write(rec); f.flush(); os.fsync(f.fileno())

    def _load_wal(self):
        if not os.path.exists(self.wal_path): return
        with open(self.wal_path, "rb") as f: data = f.read()
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
            if nid not in self.cells:
                self.cells[nid] = {"signature": vec, "context": b"\x00"*64, "links": [],
                    "gate_bias": 0.5, "access_count": 0, "memory_version": 0, "last_accessed": 0.0}
            self.total += 1

    def save(self):
        with open(os.path.join(self.data_dir, "meta.json"), "w") as f:
            json.dump({"total": self.total, "buckets": len(self.buckets),
                       "cells": len(self.cells), "n_hash_bits": self.n_hash_bits}, f, indent=2)

    def search(self, query, top_k=10):
        bid = self._hash(query)
        candidates = []
        seen = set()
        # 1-bit neighbors
        for b in [bid] + [bid ^ (1 << b) for b in range(self.n_hash_bits)]:
            bucket = self.buckets.get(b)
            if bucket:
                with self.lock:
                    for nid, vec in bucket:
                        if nid not in seen: seen.add(nid); candidates.append((nid, vec))
        # 2-bit neighbors if needed
        if len(candidates) < top_k * 10:
            for b1 in range(self.n_hash_bits):
                for b2 in range(b1+1, self.n_hash_bits):
                    b = bid ^ (1<<b1) ^ (1<<b2)
                    bucket = self.buckets.get(b)
                    if bucket:
                        with self.lock:
                            for nid, vec in bucket:
                                if nid not in seen: seen.add(nid); candidates.append((nid, vec))
                    if len(candidates) >= top_k * 30: break
                if len(candidates) >= top_k * 30: break
        if not candidates: return []
        qnorm = math.sqrt(sum(v*v for v in query))
        if qnorm == 0: return []
        q = [v/qnorm for v in query]
        results = [(sum(a*b for a,b in zip(q, vec)), nid) for nid, vec in candidates]
        results.sort(reverse=True)
        return [(nid, sim) for sim, nid in results[:top_k]]

    def get_stats(self):
        sizes = [len(v) for v in self.buckets.values()]
        return {"total": self.total, "buckets": len(self.buckets),
                "cells": len(self.cells),
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "avg": sum(sizes)/len(sizes) if sizes else 0}


# ═══════════════════════════════════════════════
# 1. write_to_active_neurons
# ═══════════════════════════════════════════════

DIM = 64
GATE_BIAS_RATE = 0.05
GATE_MIN = 0.01
GATE_MAX = 0.99
next_nid = [100_000_000]

def write_to_active(index: "LSHIndex", active: list[tuple[int, float]], 
                    input_vector: list[float], context: bytes = b"", links=None) -> int:
    ctx = (context if isinstance(context, bytes) else b"")[:64]
    ctx = (ctx + b"\x00"*64)[:64]
    now = time.time()
    updated = 0
    with index.lock:
        for nid, sim in active:
            cell = index.cells.get(nid)
            if not cell: continue
            ver = cell["memory_version"]
            alpha = 1.0 / (ver + 2.0)
            new_sig = [old*(1-alpha) + inp*alpha for old, inp in zip(cell["signature"], input_vector[:DIM])]
            norm = math.sqrt(sum(v*v for v in new_sig))
            if norm > 0: new_sig = [v/norm for v in new_sig]
            cell["signature"] = new_sig
            cell["context"] = ctx
            if links:
                exist = set(cell["links"])
                new_links = [l for l in links if l not in exist]
                cell["links"].extend(new_links)
                if len(cell["links"]) > 16: cell["links"] = cell["links"][-16:]
            cell["memory_version"] += 1
            cell["access_count"] += 1
            cell["last_accessed"] = now
            updated += 1
    return updated


def create_new_neuron(index: "LSHIndex", input_vector: list[float],
                      context: bytes = b"", links=None) -> int:
    global next_nid
    nid = next_nid[0]; next_nid[0] += 1
    sig = list(input_vector[:DIM])
    norm = math.sqrt(sum(v*v for v in sig))
    if norm > 0: sig = [v/norm for v in sig]
    index.add(nid, sig, context, links)
    return nid


def update_gate_bias(cell: dict, reward: float) -> float:
    if reward > 0: cell["gate_bias"] *= (1.0 - GATE_BIAS_RATE)
    elif reward < 0: cell["gate_bias"] *= (1.0 + GATE_BIAS_RATE)
    cell["gate_bias"] = max(GATE_MIN, min(GATE_MAX, cell["gate_bias"]))
    cell["memory_version"] += 1
    return cell["gate_bias"]


# ═══════════════════════════════════════════════
# Demo: E2E with LSH
# ═══════════════════════════════════════════════

def e2e_lsh_demo():
    import shutil
    d = os.path.expanduser("~/neuron-data/lsh_e2e")
    if os.path.exists(d): shutil.rmtree(d)

    idx = LSHIndex(data_dir=d, n_hash_bits=8, use_numpy=True)
    rng = random.Random(42)

    N = 100_000
    N_CLUSTERS = 10
    CYCLES = 10
    TOP_K = 10

    # Generate clustered data
    centroids = []
    for _ in range(N_CLUSTERS):
        c = [rng.gauss(0,1) for _ in range(64)]
        n = math.sqrt(sum(v*v for v in c))
        if n > 0: c = [v/n for v in c]
        centroids.append(c)

    print(f"Building {N:,} neurons in {N_CLUSTERS} clusters...")
    t0 = time.perf_counter()
    for i in range(N):
        centroid = centroids[i % N_CLUSTERS]
        vec = [c + rng.gauss(0,0.3) for c in centroid]
        n = math.sqrt(sum(v*v for v in vec))
        if n > 0: vec = [v/n for v in vec]
        idx.add(i, vec)
    bt = time.perf_counter() - t0
    print(f"  Built in {bt:.1f}s")
    print(f"  {idx.get_stats()}")

    # Learning cycles
    print(f"\n── Learning Cycles ({CYCLES}) ──")
    total_created = 0
    for cycle in range(CYCLES):
        # Random query (from cluster centroids + noise)
        c = centroids[cycle % N_CLUSTERS]
        qv = [x + rng.gauss(0,0.3) for x in c]
        n = math.sqrt(sum(v*v for v in qv))
        if n > 0: qv = [v/n for v in qv]

        t0 = time.perf_counter()
        active = idx.search(qv, top_k=TOP_K)
        ms = (time.perf_counter()-t0)*1000

        ctx = struct.pack("!dI", time.time(), cycle)
        links = [cycle*1000+i for i in range(3)]
        written = write_to_active(idx, active, qv, ctx, links)

        # Gate bias
        pos = neg = 0
        for nid, sim in active:
            cell = idx.cells.get(nid)
            if cell:
                update_gate_bias(cell, 1.0 if sim > 0.5 else -0.5)
                pos += 1 if sim > 0.5 else 0; neg += 1 if sim <= 0.5 else 0

        created = 0
        if len(active) < 5:
            create_new_neuron(idx, qv, ctx, links)
            total_created += 1
            created = 1

        print(f"  Cycle {cycle+1:2d}: active={len(active):2d} written={written} "
              f"pos={pos:2d}/neg={neg:2d} create={created} ({ms:.0f}ms)")

    # Stats
    cells = list(idx.cells.values())
    print(f"\n── Final Stats ──")
    print(f"  Total neurons: {idx.total:,}")
    print(f"  New neurons created: {total_created}")
    print(f"  Cells with writes: {sum(1 for c in cells if c['memory_version'] > 0)}")
    avg_ver = sum(c['memory_version'] for c in cells) / max(1, len(cells))
    print(f"  Avg memory_version: {avg_ver:.2f}")
    biases = [c['gate_bias'] for c in cells]
    print(f"  Gate bias: [{min(biases):.3f}, {max(biases):.3f}]")
    print(f"  Cells == total: {len(cells) == idx.total}")
    print(f"  Done!")


if __name__ == "__main__":
    import sys
    if "--e2e" in sys.argv:
        e2e_lsh_demo()
    elif "--demo" in sys.argv:
        # Quick LSH precision demo
        import shutil
        d = os.path.expanduser("~/neuron-data/lsh_mem_demo")
        if os.path.exists(d): shutil.rmtree(d)
        idx = LSHIndex(data_dir=d, n_hash_bits=8, use_numpy=True)
        rng = random.Random(42)
        N, NC = 100000, 10
        centroids = []
        for _ in range(NC):
            c = [rng.gauss(0,1) for _ in range(64)]
            n = math.sqrt(sum(v*v for v in c))
            if n > 0: c = [v/n for v in c]
            centroids.append(c)
        for i in range(N):
            c = centroids[i % NC]
            vec = [x + rng.gauss(0,0.3) for x in c]
            n = math.sqrt(sum(v*v for v in vec))
            if n > 0: vec = [v/n for v in vec]
            idx.add(i, vec)
        # Use a centroid as query WITH noise so it's not exactly one of the stored points
        rng2 = random.Random(42)
        qv = [x + rng2.gauss(0,0.3) for x in centroids[0]]
        n = math.sqrt(sum(v*v for v in qv))
        if n > 0: qv = [v/n for v in qv]
        
        brute = sorted([(sum(a*b for a,b in zip(qv, idx.vectors[i])), i) for i in range(N)], reverse=True)
        lsh = idx.search(qv, top_k=10)
        overlap = len(set(nid for nid,_ in lsh) & set(nid for _,nid in brute[:10]))
        print(f"LSH Precision@10: {overlap*10:.0f}%")
        print(f"Brute top-3 IDs: {[nid for _,nid in brute[:3]]}")
        print(f"LSH top-3 IDs:   {[nid for nid,_ in lsh[:3]]}")
    else:
        print("Usage: python3 lsh_memory.py --demo    (LSH precision test)")
        print("       python3 lsh_memory.py --e2e     (full online learning cycle)")
