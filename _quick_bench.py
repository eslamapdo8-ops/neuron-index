"""Quick LSH benchmark: L=4, 100K, Precision@10"""
import sys, os, math, random, time, shutil
sys.path.insert(0, os.path.expanduser("~/neuron-index"))
from new_lsh_index import LSHIndex

d = os.path.expanduser("~/neuron-data/lsh_quick")
if os.path.exists(d): shutil.rmtree(d)

idx = LSHIndex(L=4, k=12, data_dir=d)
rng = random.Random(42)

N = 100_000
N_CLUSTERS = 10
DIM = 64

# Generate centroids
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

# Bucket stats
stats = idx.get_stats()
for ti, tbl in enumerate(stats["tables"]):
    print(f"  Table {ti}: buckets={tbl['count']}, avg={tbl['avg']:.1f}, max={tbl['max']}")

# 100 queries
NQ = 100
all_prec = []
all_lat = []
all_cand = []

for q in range(NQ):
    cid = q % N_CLUSTERS
    qv = [c + rng.gauss(0, 0.2) for c in centroids[cid]]
    norm = math.sqrt(sum(v*v for v in qv))
    if norm > 0: qv = [v/norm for v in qv]

    # brute force ground truth
    brute = idx.brute_force_search(qv, top_k=10)
    brute_set = set(n for n,_,_ in brute)

    # LSH 2-stage
    t_start = time.perf_counter()
    cand = idx.lsh_lookup(qv)
    res = idx.filter_by_relevance(cand, qv, threshold=0.0, top_k=10)
    lat = (time.perf_counter() - t_start) * 1000

    res_set = set(n for n,_,_ in res)
    prec = len(brute_set & res_set) / 10.0 * 100

    all_prec.append(prec)
    all_lat.append(lat)
    all_cand.append(len(cand))

avg_prec = sum(all_prec) / len(all_prec)
avg_lat = sum(all_lat) / len(all_lat)
avg_cand = sum(all_cand) / len(all_cand)

print(f"\n{'='*50}")
print(f"RESULTS: L=4, k=12, N=100K, queries={NQ}")
print(f"{'='*50}")
print(f"  Build time:     {build_time:.2f}s")
print(f"  Avg query:      {avg_lat:.2f}ms")
print(f"  Avg candidates: {avg_cand:.0f}")
print(f"  Precision@10:   {avg_prec:.1f}%")
if avg_prec >= 90.0:
    print(f"  ✅ PASS (≥90%)")
else:
    print(f"  ❌ FAIL (<90%)")
