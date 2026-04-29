"""Quick LSH benchmark: LSH only (no brute force), measure candidates"""
import sys, os, math, random, time, shutil
sys.path.insert(0, os.path.expanduser("~/neuron-index"))
from new_lsh_index import LSHIndex

d = os.path.expanduser("~/neuron-data/lsh_quick2")
if os.path.exists(d): shutil.rmtree(d)

L_VAL = 8
idx = LSHIndex(L=L_VAL, k=12, data_dir=d)
rng = random.Random(42)

N = 100_000
N_CLUSTERS = 10
DIM = 64

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
build_time = time.perf_counter() - t0
print(f"Build: {build_time:.2f}s ({N/build_time:.0f}/s)")

stats = idx.get_stats()
for ti, tbl in enumerate(stats["tables"]):
    print(f"  T{ti}: buckets={tbl['count']}, avg={tbl['avg']:.1f}, max={tbl['max']}")

# Benchmark: measure LSH lookup + filter time, count candidates
NQ = 10
print(f"\nQuery benchmark ({NQ} queries):")
all_lat = []
all_cand = []
for q in range(NQ):
    cid = q % N_CLUSTERS
    qv = [c + rng.gauss(0, 0.2) for c in centroids[cid]]
    norm = math.sqrt(sum(v*v for v in qv))
    if norm > 0: qv = [v/norm for v in qv]

    t0 = time.perf_counter()
    cand = idx.lsh_lookup(qv)
    t1 = time.perf_counter()
    res = idx.filter_by_relevance(cand, qv, threshold=0.0, top_k=10)
    t2 = time.perf_counter()
    
    all_cand.append(len(cand))
    all_lat.append((t2 - t0) * 1000)
    top_ids = [n for n,_,_ in res[:5]]
    print(f"  Q{q}: {len(cand):>5} cand, {len(res):>2} res, {(t2-t0)*1000:.1f}ms — top: {top_ids[:3]}")

avg_lat = sum(all_lat) / len(all_lat)
avg_cand = sum(all_cand) / len(all_cand)

# Brute force only on first query for Precision comparison
print(f"\nBrute force comparison (1 query)...")
qv = [c + rng.gauss(0, 0.2) for c in centroids[0]]
norm = math.sqrt(sum(v*v for v in qv))
if norm > 0: qv = [v/norm for v in qv]

bf_start = time.perf_counter()
scored = []
for nid, cell in idx.cells.items():
    sim = idx.cells[nid].signature  # wrong, need cosine
    # Actually compute cosine
    dot = sum(a*b for a,b in zip(qv, idx.cells[nid].signature))
    na = math.sqrt(sum(v*v for v in qv))
    nb = math.sqrt(sum(v*v for v in idx.cells[nid].signature))
    sim_val = dot / (na * nb) if na*nb > 0 else 0
    scored.append((sim_val, nid))
scored.sort(reverse=True)
bf_time = (time.perf_counter() - bf_start) * 1000
bf_top10 = set(n for _, n in scored[:10])

# LSH top-10
cand = idx.lsh_lookup(qv)
res = idx.filter_by_relevance(cand, qv, threshold=0.0, top_k=10)
lsh_top10 = set(n for n,_,_ in res)
prec = len(bf_top10 & lsh_top10) / 10.0 * 100

# Recall@50
bf_top50 = set(n for _, n in scored[:50])
recall = len(lsh_top10 & bf_top50) / 10.0 * 100

print(f"\n{'='*50}")
print(f"RESULTS: L={L_VAL}, k=12, N={N:,}")
print(f"{'='*50}")
print(f"  Build time:     {build_time:.2f}s")
print(f"  Avg query:      {avg_lat:.2f}ms")
print(f"  Avg candidates: {avg_cand:.0f}")
print(f"  Brute force q:  {bf_time:.0f}ms (1 query)")
print(f"  Precision@10:   {prec:.1f}%")
print(f"  Recall@50:      {recall:.1f}%")
if prec >= 90.0:
    print(f"  ✅ PASS (≥90%)")
else:
    print(f"  ❌ FAIL (<90%)")
