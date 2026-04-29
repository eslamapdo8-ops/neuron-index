"""E2LSH-style: L=4, k=8 (256 buckets/table) — fewer buckets = better clustering"""
import sys, os, math, random, time, shutil
sys.path.insert(0, "/workspaces/neuron-index")
from new_lsh_index import LSHIndex, cosine_similarity, DIM

d = os.path.expanduser("~/neuron-data/lsh_e2lsh")
if os.path.exists(d): shutil.rmtree(d)

for L, k in [(4, 8), (8, 8), (4, 10), (2, 10)]:
    d2 = os.path.expanduser(f"~/neuron-data/lsh_L{L}_k{k}")
    if os.path.exists(d2): shutil.rmtree(d2)
    idx = LSHIndex(L=L, k=k, data_dir=d2)
    rng = random.Random(42)
    N = 100_000
    N_CLUSTERS = 10
    centroids = []
    for c in range(N_CLUSTERS):
        centroid = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v*v for v in centroid))
        if norm > 0: centroid = [v/norm for v in centroid]
        centroids.append(centroid)
    t0 = time.perf_counter()
    for i in range(N):
        cluster = i % N_CLUSTERS
        vec = [c + rng.gauss(0, 0.3) for c in centroids[cluster]]
        norm = math.sqrt(sum(v*v for v in vec))
        if norm > 0: vec = [v/norm for v in vec]
        idx.add_neuron(i, vec)
    bt = time.perf_counter() - t0
    stats = idx.get_stats()
    avg_buck = stats["tables"][0]["avg"]
    # Single query: centroid query → should match its own cluster
    qv = [c + rng.gauss(0, 0.2) for c in centroids[0]]
    norm = math.sqrt(sum(v*v for v in qv))
    if norm > 0: qv = [v/norm for v in qv]
    t0 = time.perf_counter()
    cand = idx.lsh_lookup(qv)
    res = idx.filter_by_relevance(cand, qv, threshold=0.0, top_k=10)
    lat = (time.perf_counter() - t0) * 1000
    lsh_set = set(n for n,_,_ in res)
    bf = [(cosine_similarity(qv, cell.signature), nid) for nid, cell in idx.cells.items()]
    bf.sort(reverse=True)
    bf_set = set(n for _, n in bf[:10])
    prec = len(lsh_set & bf_set) / 10.0 * 100
    print(f"L={L:2d} k={k:2d} | Build={bt:5.1f}s | Buc_avg={avg_buck:5.1f} | Cand={len(cand):>5} | Lat={lat:.1f}ms | P@10={prec:.0f}%")
