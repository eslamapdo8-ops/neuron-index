"""
e2e_lsh_demo.py — End-to-End Active Inference with Multi-Table LSH
===================================================================

Full cycle:
  1. Build LSH index with N random neurons
  2. For each learning cycle:
     a. Generate random query
     b. select_active_neurons → 2-stage (LSH lookup + filter_by_relevance)
     c. write_to_active_neurons → update their memory (moving average + context + links)
     d. update_gate_bias → reinforce successful cells based on similarity
     e. create_new_neuron if active set is too small
  3. Show final stats:
     - Total neurons (should grow if new neurons created)
     - Average memory_version (should increase)
     - Average access_count
     - Gate bias distribution

Usage:
  python3 e2e_lsh_demo.py [--fast] [--neurons=N] [--cycles=N] [--topk=N] [--threshold=T]
"""

import time
import math
import random
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from new_lsh_index import (
    LSHIndex, write_to_active_neurons, create_new_neuron,
    update_gate_bias, cosine_similarity,
    NUM_TABLES, N_HASH_BITS, DIM
)


def e2e_lsh_demo(
    n_initial: int = 100_000,
    n_cycles: int = 10,
    top_k: int = 10,
    threshold: float = 0.15,
    seed: int = 42,
    L: int = 4,        # use L=4 for speed; L=8 gives higher recall
    k: int = 8,        # k=8 (256 buckets) clusters better for 100K
):
    import shutil

    d = os.path.expanduser("~/neuron-data/e2e_lsh_demo")
    if os.path.exists(d):
        shutil.rmtree(d)

    print(f"{'=' * 60}")
    print(f"E2E LSH DEMO — Active Inference + Online Learning")
    print(f"{'=' * 60}")
    print(f"Initial neurons:  {n_initial:,}")
    print(f"Learning cycles:  {n_cycles}")
    print(f"Top-K active:     {top_k}")
    print(f"Threshold:        {threshold}")
    print(f"LSH:              L={L}, k={k} ({1 << k:,} buckets/table)")
    print()

    # ── 1. Build index ──
    print("─── Phase 1: Build Index ───")
    index = LSHIndex(L=L, k=k, data_dir=d)
    rng = random.Random(seed)

    t0 = time.perf_counter()
    for i in range(n_initial):
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        index.add_neuron(i, vec)
    build_time = time.perf_counter() - t0

    stats = index.get_stats()
    print(f"  Built {n_initial:,} neurons in {build_time:.2f}s")
    print(f"  Active buckets/table: {stats['tables'][0]['count']}")
    print(f"  Avg bucket size:      {stats['tables'][0]['avg']:.1f}")
    print()

    # ── 2. Learning cycles ──
    print("─── Phase 2: Learning Cycles ───")

    cycle_stats = []
    total_neurons_created = 0

    for cycle in range(n_cycles):
        print(f"\n  Cycle {cycle + 1}/{n_cycles} ", end="", flush=True)

        # Generate query
        qv = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(v * v for v in qv))
        if norm > 0:
            qv = [v / norm for v in qv]

        # a) Select active neurons (2-stage)
        t0 = time.perf_counter()

        # Stage 1: LSH lookup
        candidates = index.lsh_lookup(qv)

        # Stage 2: filter by relevance with gate bias
        active = index.filter_by_relevance(candidates, qv, threshold=threshold, top_k=top_k)
        select_time = time.perf_counter() - t0

        # b) Write to active neurons
        context = struct.pack("!dI", time.time(), cycle)
        new_links = [cycle * 1000 + i for i in range(3)]
        written = write_to_active_neurons(index, active, qv, context, new_links)

        # c) Update gate bias (RL)
        rewards_pos = 0
        rewards_neg = 0
        for nid, sim, _ in active:
            cell = index.cells.get(nid)
            if cell:
                # Reward: +1 if sim > threshold + 0.1 (strong match), -1 otherwise
                reward = 1.0 if sim > threshold + 0.1 else -1.0
                update_gate_bias(cell, reward)
                if reward > 0:
                    rewards_pos += 1
                else:
                    rewards_neg += 1

        # d) Create new neuron if too few active
        created = 0
        if len(active) < 3:
            new_nid = create_new_neuron(index, qv, context, new_links)
            total_neurons_created += 1
            created = 1

        # Track stats
        cycle_stats.append({
            "cycle": cycle,
            "candidates": len(candidates),
            "active_count": len(active),
            "select_time_ms": select_time * 1000,
            "written": written,
            "rewards_pos": rewards_pos,
            "rewards_neg": rewards_neg,
            "created": created,
        })

        print(f"cand={len(candidates):>4} active={len(active):>2} "
              f"written={written} rewards=+{rewards_pos}/-{rewards_neg} "
              f"created={created} ({select_time * 1000:.1f}ms)", flush=True)

    # ── 3. Final statistics ──
    print(f"\n─── Phase 3: Final Statistics ───")

    avg_cand = sum(s["candidates"] for s in cycle_stats) / len(cycle_stats)
    avg_active = sum(s["active_count"] for s in cycle_stats) / len(cycle_stats)
    avg_select = sum(s["select_time_ms"] for s in cycle_stats) / len(cycle_stats)
    total_written = sum(s["written"] for s in cycle_stats)
    total_pos = sum(s["rewards_pos"] for s in cycle_stats)
    total_neg = sum(s["rewards_neg"] for s in cycle_stats)

    cells = list(index.cells.values())
    cells_with_writes = [c for c in cells if c.memory_version > 0]
    cell_ages = [c.memory_version for c in cells if c.memory_version > 0]
    cell_access = [c.access_count for c in cells if c.access_count > 0]
    cell_biases = [c.gate_bias for c in cells]

    print(f"\n  ┌─────────────────────────────────┬──────────┐")
    print(f"  │ Metric                          │ Value    │")
    print(f"  ├─────────────────────────────────┼──────────┤")
    print(f"  │ Build time                      │ {build_time:>8.2f}s │")
    print(f"  │ Total neurons                   │ {index.total_neurons:>8,} │")
    print(f"  │ Avg candidates per cycle        │ {avg_cand:>8.0f} │")
    print(f"  │ Avg active neurons per cycle    │ {avg_active:>8.1f} │")
    print(f"  │ Avg query time                  │ {avg_select:>8.2f}ms │")
    print(f"  │ Total write operations          │ {total_written:>8,} │")
    print(f"  │ Positive rewards                │ {total_pos:>8,} │")
    print(f"  │ Negative rewards                │ {total_neg:>8,} │")
    print(f"  │ New neurons created             │ {total_neurons_created:>8,} │")
    print(f"  │ Cells with writes               │ {len(cells_with_writes):>8,} │")
    print(f"  │ Max memory_version              │ {max(cell_ages) if cell_ages else 0:>8,} │")
    print(f"  │ Avg memory_version (written)    │ {sum(cell_ages)/len(cell_ages):>8.1f}" if cell_ages else "  │ ...")
    print(f"  │ Avg access_count  (written)     │ {sum(cell_access)/len(cell_access):>8.1f}" if cell_access else "  │ ...")
    print(f"  │ Gate bias range                 │ [{min(cell_biases):.3f}, {max(cell_biases):.3f}] │")
    print(f"  └─────────────────────────────────┴──────────┘")

    # Integrity checks
    version_total = sum(c.memory_version for c in cells)
    access_total = sum(c.access_count for c in cells)
    print(f"\n  Integrity checks:")
    print(f"    Total memory writes recorded: {version_total:,}")
    print(f"    Total access counts:          {access_total:,}")
    print(f"    cells == total_neurons:       {len(cells) == index.total_neurons}")

    print(f"\n{'=' * 60}")
    print(f"E2E LSH DEMO COMPLETE — LSH-based learning works!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    n_initial = 100_000
    n_cycles = 10
    top_k = 10
    threshold = 0.15

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--fast":
            n_initial = 10_000
            n_cycles = 5
            threshold = 0.1
        elif arg.startswith("--neurons="):
            n_initial = int(arg.split("=")[1])
        elif arg.startswith("--cycles="):
            n_cycles = int(arg.split("=")[1])
        elif arg.startswith("--topk="):
            top_k = int(arg.split("=")[1])
        elif arg.startswith("--threshold="):
            threshold = float(arg.split("=")[1])
        elif arg == "--help":
            print("Usage: python3 e2e_lsh_demo.py [options]")
            print("  --fast           Quick demo (10K neurons, 5 cycles)")
            print("  --neurons=N      Initial neuron count (default: 100K)")
            print("  --cycles=N       Learning cycles (default: 10)")
            print("  --topk=N         Active neurons per query (default: 10)")
            print("  --threshold=T    Similarity threshold (default: 0.15)")
            sys.exit(0)

    e2e_lsh_demo(
        n_initial=n_initial,
        n_cycles=n_cycles,
        top_k=top_k,
        threshold=threshold,
    )
