# Neuron Index — LSH-based Active Inference

## Phase 1 ✅ — Multi-Table LSH

**LSH with Random Projections (L=8, k=8) achieves Precision@10 = 100% on 100K clustered data.**

| Config | Precision@10 | Candidates | Query Time | Notes |
|--------|-------------|-----------|------------|-------|
| L=8, k=8 | **100%** | ~30K | 234ms | Highest precision |
| L=4, k=8 | 80% | ~16K | 113ms | Balanced |
| L=4, k=10 | 70% | ~5K | 47ms | Fast query |
| L=2, k=10 | 60% | ~2.4K | 20ms | Lowest candidates |

## Architecture

```
Layer 1: LSH Multi-Table (L tables, k-bit random Gaussian projections)
  ↓ candidate list (union of all table matches)
Layer 2: filter_by_relevance (cosine similarity + gate_bias threshold)
  ↓ active neurons
Online Learning: write_to_active_neurons + update_gate_bias + create_new_neuron
```

## Files

| File | Description |
|------|-------------|
| `new_lsh_index.py` | Multi-table LSH + CellState + 2-stage selection + write mechanism |
| `e2e_lsh_demo.py` | End-to-end learning demo (10 cycles) |
| `linear_index.py` | Flat brute-force (100% Precision, used as ground truth) |
| `memory_write.py` | Online learning write mechanism (standalone) |
| `neuron_index.py` | HNSW-based index (legacy) |
| `e2e_demo.py` | Flat-index e2e demo (legacy) |

## Next Steps

- **Phase 2:** Real Gating (activate only 0.1% of neurons)
- **Phase 3:** 100M-scale stress test with FAISS IVF
- **Phase 4:** Replace with FAISS if needed
