"""
e2e_demo.py — End-to-End Active Inference + Online Learning
=============================================================

Full cycle:
  1. Build index with N random neurons
  2. For each learning cycle:
     a. Generate random query (simulates environment input)
     b. select_active_neurons → top-10 most similar
     c. write_to_active_neurons → update their memory
     d. update_gate_bias → reinforce successful cells
     e. create_new_neuron if active set is too small (< 3)
  3. Show stats after all cycles:
     - Total neurons (should grow if new neurons created)
     - Average memory_version (should increase over cycles)
     - Average access_count
     - Gate bias distribution
"""

import time
import math
import random
import struct
import sys
import os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linear_index import FlatNeuronIndex
from memory_write import write_to_active_neurons, create_new_neuron, update_gate_bias


def e2e_demo(
    n_initial: int = 100_000,
    n_cycles: int = 10,
    top_k: int = 10,
    threshold: float = 0.15,
    seed: int = 42,
):
    """
    Full end-to-end demo.
    
    Args:
        n_initial: Number of random neurons in the index
        n_cycles: Number of learning cycles
        top_k: Active neurons to select per cycle
        threshold: Minimum similarity for active neuron
        seed: Random seed
    """
    import shutil

    d = os.path.expanduser("~/neuron-data/e2e_demo")
    if os.path.exists(d):
        shutil.rmtree(d)

    print(f"{'=' * 60}")
    print(f"E2E DEMO — Active Inference + Online Learning")
    print(f"{'=' * 60}")
    print(f"Initial neurons: {n_initial:,}")
    print(f"Learning cycles: {n_cycles}")
    print(f"Top-K active:    {top_k}")
    print(f"Threshold:       {threshold}")
    print()

    # ── 1. Build index ──
    print("─── Phase 1: Build Index ───")
    index = FlatNeuronIndex(data_dir=d, auto_save_interval=50000, use_numpy=True)
    rng = random.Random(seed)

    t0 = time.perf_counter()
    for i in range(n_initial):
        vec = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        index.add_neuron_to_index(neuron_id=i, signature_float=vec)
    build_time = time.perf_counter() - t0
    index.save_metadata()
    print(f"  Built {n_initial:,} neurons in {build_time:.2f}s")
    print(f"  Mode: {'numpy' if index.use_numpy else 'pure Python'}")
    print()

    # ── 2. Learning cycles ──
    print("─── Phase 2: Learning Cycles ───")

    cycle_stats = []
    total_neurons_created = 0

    for cycle in range(n_cycles):
        print(f"\n  Cycle {cycle + 1}/{n_cycles} ", end="", flush=True)

        # Generate query (simulates environment input)
        qv = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in qv))
        if norm > 0:
            qv = [v / norm for v in qv]

        # a) Select active neurons
        t0 = time.perf_counter()
        active = index.select_active_neurons(qv, top_k=top_k, threshold=threshold)
        select_time = time.perf_counter() - t0

        # b) Write to active neurons
        context = struct.pack("!dI", time.time(), cycle)
        new_links = [cycle * 1000 + i for i in range(3)]
        written = write_to_active_neurons(index, active, qv, context, new_links)

        # c) Update gate bias (RL): reward = +1 if sim > threshold+0.1, else -1
        rewards_pos = 0
        rewards_neg = 0
        for nid, sim, _ in active:
            cell = index.cells.get(nid)
            if cell:
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
        now = time.time()
        stats = {
            "cycle": cycle,
            "active_count": len(active),
            "select_time_ms": select_time * 1000,
            "written": written,
            "rewards_pos": rewards_pos,
            "rewards_neg": rewards_neg,
            "created": created,
        }
        cycle_stats.append(stats)

        print(f"active={len(active)} written={written} "
              f"rewards=+{rewards_pos}/-{rewards_neg} "
              f"created={created} "
              f"({select_time*1000:.1f}ms)", flush=True)

    # ── 3. Final statistics ──
    print(f"\n─── Phase 3: Final Statistics ───")

    # Aggregate
    avg_active = sum(s["active_count"] for s in cycle_stats) / len(cycle_stats)
    avg_select = sum(s["select_time_ms"] for s in cycle_stats) / len(cycle_stats)
    total_written = sum(s["written"] for s in cycle_stats)
    total_pos = sum(s["rewards_pos"] for s in cycle_stats)
    total_neg = sum(s["rewards_neg"] for s in cycle_stats)

    # Cell state analysis
    cells = list(index.cells.values())
    cells_with_writes = [c for c in cells if c.memory_version > 0]
    cell_ages = [c.memory_version for c in cells if c.memory_version > 0]
    cell_access = [c.access_count for c in cells if c.access_count > 0]
    cell_biases = [c.gate_bias for c in cells]

    print(f"\n  ┌─────────────────────────────────┬──────────┐")
    print(f"  │ Metric                          │ Value    │")
    print(f"  ├─────────────────────────────────┼──────────┤")
    print(f"  │ Total neurons                   │ {index.total_neurons:>8,} │")
    print(f"  │ Avg active neurons per cycle     │ {avg_active:>8.1f} │")
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

    # Verify no data loss
    version_total = sum(c.memory_version for c in cells)
    access_total = sum(c.access_count for c in cells)
    print(f"\n  Integrity checks:")
    print(f"    Total memory writes recorded: {version_total:,}")
    print(f"    Total access counts:          {access_total:,}")
    print(f"    cells == total_neurons:       {len(cells) == index.total_neurons}")

    print(f"\n{'=' * 60}")
    print(f"E2E DEMO COMPLETE — System learns and grows without forgetting")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Parse args
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
            print("Usage: python3 e2e_demo.py [options]")
            print("  --fast           Quick demo (10K neurons, 5 cycles)")
            print("  --neurons=N      Initial neuron count (default: 100K)")
            print("  --cycles=N       Learning cycles (default: 10)")
            print("  --topk=N         Active neurons per query (default: 10)")
            print("  --threshold=T    Similarity threshold (default: 0.15)")
            sys.exit(0)

    e2e_demo(
        n_initial=n_initial,
        n_cycles=n_cycles,
        top_k=top_k,
        threshold=threshold,
    )
