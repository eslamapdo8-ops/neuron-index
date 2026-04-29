"""Test LSH with L=1, k=16 (65536 buckets) for Precision@10"""
import sys, os, math, random, time, shutil
sys.path.insert(0, "/workspaces/neuron-index")
from new_lsh_index import LSHIndex, cosine_similarity, DIM

d = os.path.expanduser("~/neuron-data/lsh_tune")
if os.path.exists(d): shutil.rmtree(d)

idx = LSHIndex(L=1, k=16, data_dir=d)
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
build_time = time.perf_counter() - t0
print(f"Build: {build_time:.2f}s ({N/build_time:.0f}/s)")

stats = idx.get_stats()
for ti, tbl in enumerate(stats["tables"]):
    print(f"  T{ti}: buckets={tbl['count']}, avg={tbl['avg']:.1f}, max={tbl['max']}")

print(f"\nQuery benchmark:")
all_prec, all_cand, all_lat = [], [], []
for q in range(10):
    cid = q % N_CLUSTERS
    qv = [c + rng.gauss(0, 0.2) for c in centroids[cid]]
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
    bf50 = set(n for _, n in bf[:50])
    prec = len(lsh_set & bf_set) / 10.0 * 100
    recall = len(lsh_set & bf50) / 10.0 * 100
    all_prec.append(prec); all_cand.append(len(cand)); all_lat.append(lat)
    print(f"  Q{q}: cand={len(cand):>4} lat={lat:.1f}ms prec={prec:.0f}% recall={recall:.0f}%")

avg_prec = sum(all_prec)/len(all_prec)
avg_cand = sum(all_cand)/len(all_cand)
avg_lat = sum(all_lat)/len(all_lat)
print(f"\nResults: L=1, k=16, N=100K | Build={build_time:.1f}s | Query={avg_lat:.1f}ms | Cand={avg_cand:.0f} | P@10={avg_prec:.1f}% | {'PASS' if avg_prec>=90 else 'FAIL'}")
