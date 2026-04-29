"""
memory_write.py — Online Learning Write Mechanism for Neuron Index
====================================================================

Three core functions:

1. write_to_active_neurons(active_neurons, input_vector, context_data, link_ids)
   - For each active neuron:
     - Moving average: new_sig = (old_sig * memory_version + input) / (memory_version + 1)
     - Write context to buffer (Bytes 64-127)
     - Update link table (Bytes 128-191, FIFO per cell)
     - Increment memory_version and access_count

2. create_new_neuron(index, input_vector, context_data, link_ids)
   - Creates a new neuron if active set is too small
   - Calls index.add_neuron_to_index with full state

3. update_gate_bias(cell, reward_signal)
   - reward=+1: gate_bias *= 0.95 (lower = easier to activate)
   - reward=-1: gate_bias *= 1.05 (raise = harder to activate)
   - Clamped to [0.01, 0.99]

Usage:
  from memory_write import write_to_active_neurons, create_new_neuron, update_gate_bias
  index = FlatNeuronIndex()
  active = index.select_active_neurons(query_float, top_k=10, threshold=0.3)
  write_to_active_neurons(index, active, new_input, context_data, link_ids=[42, 99])
  for nid, sim, _ in active:
      if sim > 0.4:
          update_gate_bias(index.cells[nid], +1)
"""

import struct
import math
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linear_index import FlatNeuronIndex, CellState


# ═══════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════

DIMENSION = 64              # signature vector dimension
CONTEXT_SIZE = 64           # bytes
MAX_LINKS = 16              # max link IDs per cell
GATE_BIAS_LEARN_RATE = 0.05  # per reward signal
GATE_BIAS_MIN = 0.01
GATE_BIAS_MAX = 0.99


# ═══════════════════════════════════════════════
# 1. write_to_active_neurons
# ═══════════════════════════════════════════════

def write_to_active_neurons(
    index: "FlatNeuronIndex",
    active_neurons: list[tuple[int, float, int]],
    input_vector: list[float],
    context_data: bytes = b"",
    link_ids: list[int] | None = None,
) -> int:
    """
    Write new information to all active neurons.
    
    For each active neuron:
      1. Update signature via online moving average
      2. Update context buffer
      3. Update link table (FIFO)
      4. Increment memory_version and access_count
    
    Args:
        index: FlatNeuronIndex instance
        active_neurons: list from select_active_neurons — [(id, sim, hdist), ...]
        input_vector: 64-d float vector (new signature component)
        context_data: raw bytes to write to context buffer
        link_ids: neuron IDs to add to link table
    
    Returns:
        Number of neurons updated
    """
    ctx = (context_data if isinstance(context_data, bytes) else b"")[:CONTEXT_SIZE]
    ctx_padded = (ctx + b"\x00" * CONTEXT_SIZE)[:CONTEXT_SIZE]
    now = time.time()
    updated = 0

    with index.lock:
        for neuron_id, similarity, _ in active_neurons:
            cell = index.cells.get(neuron_id)
            if cell is None:
                continue

            # 1. Update signature via moving average
            ver = cell.memory_version
            alpha = 1.0 / (ver + 2.0)  # weight of new input
            new_sig = []
            for old_val, inp_val in zip(cell.signature, input_vector[:DIMENSION]):
                new_val = old_val * (1.0 - alpha) + inp_val * alpha
                new_sig.append(new_val)
            cell.signature = new_sig

            # Re-normalize to unit sphere
            norm = math.sqrt(sum(v * v for v in new_sig))
            if norm > 0:
                cell.signature = [v / norm for v in new_sig]

            # Update the vector in FlatNeuronIndex.vectors
            idx = index._id_to_index.get(neuron_id)
            if idx is not None:
                index.vectors[idx] = tuple(cell.signature)

            # 2. Update context buffer
            cell.context = ctx_padded

            # 3. Update link table (FIFO)
            if link_ids:
                existing = set(cell.links)
                new_links = [lid for lid in link_ids if lid not in existing]
                cell.links.extend(new_links)
                if len(cell.links) > MAX_LINKS:
                    cell.links = cell.links[-MAX_LINKS:]

            # 4. Increment version and access count
            cell.memory_version += 1
            cell.access_count += 1
            cell.last_accessed = now
            updated += 1

    # Invalidate numpy cache
    if index.use_numpy:
        index._np_vectors = None

    return updated


# ═══════════════════════════════════════════════
# 2. create_new_neuron
# ═══════════════════════════════════════════════

_next_neuron_id = [100_000_000]  # starts above random IDs


def create_new_neuron(
    index: "FlatNeuronIndex",
    input_vector: list[float],
    context_data: bytes = b"",
    link_ids: list[int] | None = None,
) -> int:
    """
    Create a new neuron with the given input vector as its signature.
    
    The neuron gets a unique ID (auto-incrementing from 100,000,000).
    It's added to the index and gets a CellState with default gate_bias.
    
    Args:
        index: FlatNeuronIndex instance
        input_vector: 64-d float vector
        context_data: raw bytes for context buffer
        link_ids: initial link IDs
    
    Returns:
        neuron_id of the newly created cell
    """
    global _next_neuron_id
    neuron_id = _next_neuron_id[0]
    _next_neuron_id[0] += 1

    # Normalize input
    sig = list(input_vector[:DIMENSION])
    norm = math.sqrt(sum(v * v for v in sig))
    if norm > 0:
        sig = [v / norm for v in sig]
    else:
        sig = [0.0] * DIMENSION

    index.add_neuron_to_index(
        neuron_id=neuron_id,
        signature_float=sig,
        context=context_data,
        links=link_ids,
    )

    # Ensure CellState exists (add_neuron_to_index creates it)
    if neuron_id not in index.cells:
        from linear_index import CellState
        ctx = (context_data if isinstance(context_data, bytes) else b"")[:CONTEXT_SIZE]
        ctx_padded = (ctx + b"\x00" * CONTEXT_SIZE)[:CONTEXT_SIZE]
        cell = CellState(neuron_id, sig, ctx_padded, link_ids)
        cell.gate_bias = 0.5
        index.cells[neuron_id] = cell

    return neuron_id


# ═══════════════════════════════════════════════
# 3. update_gate_bias
# ═══════════════════════════════════════════════

def update_gate_bias(
    cell: "CellState",
    reward_signal: float,
) -> float:
    """
    Update gate_bias based on reinforcement signal.
    
    reward_signal > 0 (success):
        gate_bias *= (1 - GATE_BIAS_LEARN_RATE)
        → lower bias = easier to activate next time
    
    reward_signal < 0 (failure):
        gate_bias *= (1 + GATE_BIAS_LEARN_RATE)
        → higher bias = harder to activate next time
    
    Clamped to [GATE_BIAS_MIN, GATE_BIAS_MAX].
    
    Args:
        cell: CellState instance
        reward_signal: float, positive = success, negative = failure
    
    Returns:
        New gate_bias value
    """
    if reward_signal > 0:
        cell.gate_bias *= (1.0 - GATE_BIAS_LEARN_RATE)
    elif reward_signal < 0:
        cell.gate_bias *= (1.0 + GATE_BIAS_LEARN_RATE)
    
    # Clamp
    cell.gate_bias = max(GATE_BIAS_MIN, min(GATE_BIAS_MAX, cell.gate_bias))
    cell.memory_version += 1
    
    return cell.gate_bias


# ═══════════════════════════════════════════════
# 4. Demo: Online Learning Cycle
# ═══════════════════════════════════════════════

def demo_learning_cycle():
    """
    Quick demo: index 10K neurons, run 3 learning cycles.
    Shows write, create, and gate bias updates.
    """
    from linear_index import FlatNeuronIndex
    import shutil

    d = os.path.expanduser("~/neuron-data/write_demo")
    if os.path.exists(d):
        shutil.rmtree(d)

    index = FlatNeuronIndex(data_dir=d, auto_save_interval=50000, use_numpy=True)
    rng = random.Random(42)

    # Build 10K neurons
    print("Building 10,000 neurons...")
    for i in range(10_000):
        vec = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        index.add_neuron_to_index(neuron_id=i, signature_float=vec)
    index.save_metadata()
    print(f"  Total: {index.total_neurons:,} neurons\n")

    # Learning cycles
    N_CYCLES = 3
    for cycle in range(N_CYCLES):
        print(f"─── Cycle {cycle + 1} ───")

        # Generate query
        qv = [rng.gauss(0, 1) for _ in range(64)]
        norm = math.sqrt(sum(v * v for v in qv))
        if norm > 0:
            qv = [v / norm for v in qv]

        # Step 1: Select active neurons
        t0 = time.perf_counter()
        active = index.select_active_neurons(qv, top_k=10, threshold=0.15)
        select_ms = (time.perf_counter() - t0) * 1000
        print(f"  1. select_active_neurons: {len(active)} active ({select_ms:.2f}ms)")

        # Step 2: Write to active neurons
        context = struct.pack("!d", time.time())  # timestamp as context
        new_links = [cycle * 1000 + i for i in range(5)]
        written = write_to_active_neurons(index, active, qv, context, new_links)
        print(f"  2. write_to_active_neurons: {written} neurons updated")

        # Step 3: Update gate bias (simulate random rewards)
        rewards = 0
        for nid, sim, _ in active[:5]:
            cell = index.cells.get(nid)
            if cell:
                reward = 1.0 if sim > 0.2 else -1.0
                old_bias = cell.gate_bias
                new_bias = update_gate_bias(cell, reward)
                rewards += 1
        print(f"  3. update_gate_bias: {rewards} neurons updated")

        # Step 4: Create new neuron if too few active
        if len(active) < 3:
            new_nid = create_new_neuron(index, qv, context, new_links)
            print(f"  4. create_new_neuron: [{new_nid}] created")
        else:
            print(f"  4. create_new_neuron: skipped (enough active)")

        # Show stats
        cell_sample = index.cells.get(active[0][0]) if active else None
        if cell_sample:
            print(f"     Sample cell [{cell_sample.neuron_id}]: "
                  f"ver={cell_sample.memory_version}, "
                  f"acc={cell_sample.access_count}, "
                  f"bias={cell_sample.gate_bias:.3f}, "
                  f"links={len(cell_sample.links)}")
        print()

    print("─── Final Stats ───")
    print(f"Total neurons: {index.total_neurons:,}")
    print(f"Cells with writes: {sum(1 for c in index.cells.values() if c.memory_version > 0)}")
    print(f"Avg access_count: {sum(c.access_count for c in index.cells.values()) / max(1, len(index.cells)):.1f}")
    print("Done!")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo_learning_cycle()
    else:
        print("memory_write.py — Online Learning Write Mechanism")
        print("Usage:")
        print("  python3 memory_write.py --demo    Run learning cycle demo")
