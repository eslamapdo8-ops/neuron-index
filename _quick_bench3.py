"""Ultra quick benchmark: L=2, 100K, measure Precision@10 first"""
import sys, os, math, random, time, shutil
sys.path.insert(0, os.path.expanduser("~/neuron-index"))
from new_lsh_index import LSHIndex, cosine_similarity

d = os.path.expanduser("~/neuron-data/lsh_quick3")
if os.path.exists(d): shutil.rmtree(d)

L_VAL = 2
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
    if (i+1) % 20000 == 0:
        print(f"  Built {i+1:,}/{N:,} ({time.perf_counter()-t0:.1f}s)")
build_time = time.perf_counter() - t0
print(f"✅ Build complete: {build_time:.2f}s ({N/build_time:.0f}/s)")

# Query: 10 queries, measure candidates + precision
print(f"\nQuerying...")
all_cand = []
all_lat = []
for q in range(10):
    cid = q % N_CLUSTERS
    qv = [c + rng.gauss(0, 0.2) for c in centroids[cid]]
    norm = math.sqrt(sum(v*v for v in qv))
    if norm > 0: qv = [v/norm for v in qv]
    
    t0 = time.perf_counter()
    cand = idx.lsh_lookup(qv)
    res = idx.filter_by_relevance(cand, qv, threshold=0.0, top_k=10)
    lat = (time.perf_counter() - t0) * 1000
    all_cand.append(len(cand))
    all_lat.append(lat)
    print(f"  Q{q}: {len(cand):>5} cand → {len(res):>2} res ({lat:.1f}ms)")

# Brute force on first query for Precision
print(f"\nPrecision@10 check (1 query)...")
qv = [c + rng.gauss(0, 0.2) for c in centroids[0]]
norm = math.sqrt(sum(v*v for v in qv))
if norm > 0: qv = [v/norm for v in qv]

# Brute force (only once)
cand = idx.lsh_lookup(qv)
res = idx.filter_by_relevance(cand, qv, threshold=0.0, top_k=10)
lsh_top10 = set(n for n,_,_ in res)

# Full brute force
t0 = time.perf_counter()
all_sims = []
for nid, cell in idx.cells.items():
    sim = cosine_similarity(qv, cell.signature)
    all_sims.append((sim, nid))
all_sims.sort(reverse=True)
bf_time = (time.perf_counter() - t0) * 1000
bf_top10 = set(n for _, n in all_sims[:10])
bf_top50 = set(n for _, n in all_sims[:50])

prec = len(bf_top10 & lsh_top10) / 10.0 * 100
recall = len(lsh_top10 & bf_top50) / 10.0 * 100 if lsh_top10 else 0

print(f"\n{'='*50}")
print(f"RESULTS: L={L_VAL}, k=12, N={N:,}")
print(f"{'='*50}")
print(f"  Build time:     {build_time:.2f}s")
print(f"  Avg query:      {sum(all_lat)/len(all_lat):.2f}ms")
print(f"  Avg candidates: {sum(all_cand)/len(all_cand):.0f}")
print(f"  Brute force:    {bf_time:.0f}ms")
print(f"  Precision@10:   {prec:.1f}%")
print(f"  Recall@50:      {recall:.1f}%")
if prec >= 90.0:
    print(f"  ✅ PASS (≥90%)")
else:
    print(f"  ❌ FAIL (<90%)")
