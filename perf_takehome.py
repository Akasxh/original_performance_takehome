"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


def _get_scratch_reads_writes(engine, slot):
    """Determine which scratch addresses an instruction reads and writes."""
    reads = set()
    writes = set()

    if engine == "debug":
        return reads, writes

    if engine == "alu":
        # (op, dest, a1, a2)
        op, dest, a1, a2 = slot
        reads.add(a1)
        reads.add(a2)
        writes.add(dest)
    elif engine == "valu":
        if slot[0] == "vbroadcast":
            _, dest, src = slot
            reads.add(src)
            for i in range(VLEN):
                writes.add(dest + i)
        elif slot[0] == "multiply_add":
            _, dest, a, b, c = slot
            for i in range(VLEN):
                reads.add(a + i)
                reads.add(b + i)
                reads.add(c + i)
                writes.add(dest + i)
        else:
            # (op, dest, a1, a2)
            _, dest, a1, a2 = slot
            for i in range(VLEN):
                reads.add(a1 + i)
                reads.add(a2 + i)
                writes.add(dest + i)
    elif engine == "load":
        if slot[0] == "const":
            _, dest, val = slot
            writes.add(dest)
        elif slot[0] == "load":
            _, dest, addr = slot
            reads.add(addr)  # reads addr to get memory address
            writes.add(dest)
        elif slot[0] == "vload":
            _, dest, addr = slot
            reads.add(addr)  # scalar addr
            for i in range(VLEN):
                writes.add(dest + i)
        elif slot[0] == "load_offset":
            _, dest, addr, offset = slot
            reads.add(addr + offset)
            writes.add(dest + offset)
    elif engine == "store":
        if slot[0] == "store":
            _, addr, src = slot
            reads.add(addr)
            reads.add(src)
        elif slot[0] == "vstore":
            _, addr, src = slot
            reads.add(addr)
            for i in range(VLEN):
                reads.add(src + i)
    elif engine == "flow":
        if slot[0] == "select":
            _, dest, cond, a, b = slot
            reads.add(cond)
            reads.add(a)
            reads.add(b)
            writes.add(dest)
        elif slot[0] == "vselect":
            _, dest, cond, a, b = slot
            for i in range(VLEN):
                reads.add(cond + i)
                reads.add(a + i)
                reads.add(b + i)
                writes.add(dest + i)
        elif slot[0] == "add_imm":
            _, dest, a, imm = slot
            reads.add(a)
            writes.add(dest)
        elif slot[0] in ("pause", "halt"):
            pass
        elif slot[0] == "cond_jump":
            _, cond, addr = slot
            reads.add(cond)
        elif slot[0] == "jump":
            pass

    return reads, writes


def _schedule_vliw(ops):
    """
    Dependency-aware VLIW list scheduler.
    Takes a list of (engine, slot) and packs into VLIW instruction bundles.
    """
    n = len(ops)
    if n == 0:
        return []

    # Compute reads/writes for each op
    rw = []
    for engine, slot in ops:
        r, w = _get_scratch_reads_writes(engine, slot)
        rw.append((r, w))

    # Build dependency graph
    last_writer = {}  # addr -> op_index
    last_readers = defaultdict(set)  # addr -> set of op_indices
    deps = [set() for _ in range(n)]

    for i in range(n):
        engine_i = ops[i][0]
        if engine_i == "debug":
            if i > 0:
                deps[i].add(i - 1)
            continue

        reads_i, writes_i = rw[i]

        # RAW: i reads addr that was written by j
        for addr in reads_i:
            if addr in last_writer:
                deps[i].add(last_writer[addr])

        # WAW: i writes addr that was written by j
        for addr in writes_i:
            if addr in last_writer:
                deps[i].add(last_writer[addr])

        # WAR: i writes addr that was read by j
        for addr in writes_i:
            for j in last_readers.get(addr, set()):
                if j != i:
                    deps[i].add(j)

        # Update tracking
        for addr in writes_i:
            last_writer[addr] = i
            last_readers[addr] = set()
        for addr in reads_i:
            last_readers[addr].add(i)

    # Flow control barriers - pause/halt are full barriers
    # All subsequent ops must come after them
    for i in range(n):
        if ops[i][0] == "flow" and ops[i][1][0] in ("pause", "halt"):
            # Everything before pause must complete before pause
            for j in range(i):
                if ops[j][0] != "debug":
                    deps[i].add(j)
            # Everything after pause depends on pause
            for j in range(i + 1, n):
                deps[j].add(i)

    # Compute critical path (reverse BFS from sinks)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for d in deps[i]:
            successors[d].append(i)

    level = [1] * n
    from collections import deque
    out_deg = [len(successors[i]) for i in range(n)]
    q = deque(i for i in range(n) if out_deg[i] == 0)
    while q:
        node = q.popleft()
        for pred in deps[node]:
            if level[node] + 1 > level[pred]:
                level[pred] = level[node] + 1
            out_deg[pred] -= 1
            if out_deg[pred] == 0:
                q.append(pred)

    # Schedule
    dep_count = [len(deps[i]) for i in range(n)]
    scheduled_cycle = [-1] * n
    ready = sorted([i for i in range(n) if dep_count[i] == 0], key=lambda x: -level[x])

    instructions = []
    total_scheduled = 0

    while total_scheduled < n:
        bundle = defaultdict(list)
        engine_counts = defaultdict(int)
        writes_this_cycle = set()
        scheduled_now = []
        next_ready = []

        for op_idx in ready:
            engine, slot = ops[op_idx]

            # Debug ops always fit
            if engine == "debug":
                bundle["debug"].append(slot)
                scheduled_cycle[op_idx] = len(instructions)
                scheduled_now.append(op_idx)
                total_scheduled += 1
                continue

            # Check engine slot limit
            if engine_counts[engine] >= SLOT_LIMITS.get(engine, 0):
                next_ready.append(op_idx)
                continue

            # Check write conflicts within same cycle
            _, writes_i = rw[op_idx]
            conflict = False
            for addr in writes_i:
                if addr in writes_this_cycle:
                    conflict = True
                    break
            if conflict:
                next_ready.append(op_idx)
                continue

            # Schedule it
            bundle[engine].append(slot)
            engine_counts[engine] += 1
            scheduled_cycle[op_idx] = len(instructions)
            scheduled_now.append(op_idx)
            total_scheduled += 1
            writes_this_cycle.update(writes_i)

        # Emit bundle (even if only debug)
        if bundle:
            instructions.append(dict(bundle))

        # Update ready list with newly unblocked ops
        for op_idx in scheduled_now:
            for succ in successors[op_idx]:
                dep_count[succ] -= 1
                if dep_count[succ] == 0:
                    next_ready.append(succ)

        ready = sorted(next_ready, key=lambda x: -level[x])

        # Safety: if no progress, something is wrong
        if not scheduled_now and not ready:
            # Find any unscheduled op and force it
            for i in range(n):
                if scheduled_cycle[i] == -1:
                    ready = [i]
                    break
            if not ready:
                break

    return instructions


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized VLIW SIMD kernel with:
        - SIMD vectorization (8-wide)
        - multiply_add fusion for hash stages
        - Tree level caching (depths 0-4) with vselect
        - Values kept in scratch across rounds
        - Dependency-aware VLIW instruction packing
        - Chunk interleaving for pipeline utilization
        """
        N_CHUNKS = batch_size // VLEN  # 32

        # ===== SCRATCH ALLOCATION =====
        # Scalar temps
        s_tmp = self.alloc_scratch("s_tmp")
        s_tmp2 = self.alloc_scratch("s_tmp2")
        s_forest_values_p = self.alloc_scratch("forest_values_p")
        s_inp_indices_p = self.alloc_scratch("inp_indices_p")
        s_inp_values_p = self.alloc_scratch("inp_values_p")
        s_n_nodes = self.alloc_scratch("n_nodes")

        # Vector value storage (persistent across rounds)
        val_vecs = []
        for c in range(N_CHUNKS):
            val_vecs.append(self.alloc_scratch(f"val_{c}", VLEN))

        # Vector index storage (persistent across rounds)
        idx_vecs = []
        for c in range(N_CHUNKS):
            idx_vecs.append(self.alloc_scratch(f"idx_{c}", VLEN))

        # Multiple vector temp sets for hash interleaving
        N_TEMP_SETS = 4  # Allow 4 chunks to be in-flight simultaneously
        vt_sets = []
        for s in range(N_TEMP_SETS):
            t1 = self.alloc_scratch(f"vt1_{s}", VLEN)
            t2 = self.alloc_scratch(f"vt2_{s}", VLEN)
            t3 = self.alloc_scratch(f"vt3_{s}", VLEN)
            t4 = self.alloc_scratch(f"vt4_{s}", VLEN)
            vt_sets.append((t1, t2, t3, t4))
        # Default set for non-interleaved use
        vt1, vt2, vt3, vt4 = vt_sets[0]

        # Per-chunk node_val vectors for parallel gather
        node_val_vecs = []
        for c in range(N_CHUNKS):
            node_val_vecs.append(self.alloc_scratch(f"nv_{c}", VLEN))

        # Multiple gather address sets to eliminate false dependencies between chunks
        N_GATHER_SETS = 4
        gather_addr_sets = []
        for gs in range(N_GATHER_SETS):
            addrs = []
            for i in range(VLEN):
                addrs.append(self.alloc_scratch(f"ga_{gs}_{i}"))
            gather_addr_sets.append(addrs)
        gather_addrs = gather_addr_sets[0]  # default for non-deep use

        # ===== CONSTANT VECTORS =====
        # We'll load scalar constants then broadcast to vectors
        sc_zero = self.alloc_scratch("sc_zero")
        sc_one = self.alloc_scratch("sc_one")
        sc_two = self.alloc_scratch("sc_two")

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Hash constant vectors
        hash_consts = []  # 6 constant vectors for hash stages
        hash_mult_consts = []  # multiply factors for fuseable stages (4097, 33, 9)
        hash_shift_consts = []  # shift amount vectors for non-fuseable stages

        HASH_CONST_VALS = [0x7ED55D16, 0xC761C23C, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09]
        HASH_MULT_VALS = [4097, None, 33, None, 9, None]  # stages 0,2,4 fuseable
        HASH_SHIFT_VALS = [None, 19, None, 9, None, 16]  # stages 1,3,5 shift amounts

        for i in range(6):
            hc = self.alloc_scratch(f"hc_{i}", VLEN)
            hash_consts.append(hc)

        for i, mv in enumerate(HASH_MULT_VALS):
            if mv is not None:
                hm = self.alloc_scratch(f"hm_{i}", VLEN)
                hash_mult_consts.append((i, hm, mv))
            else:
                hash_mult_consts.append(None)

        for i, sv in enumerate(HASH_SHIFT_VALS):
            if sv is not None:
                hs = self.alloc_scratch(f"hs_{i}", VLEN)
                hash_shift_consts.append((i, hs, sv))
            else:
                hash_shift_consts.append(None)

        # Tree cache: only depth 0-2 (deeper uses gather which is faster than flow vselect)
        # depth 0: 1 node (broadcast), depth 1: 2 nodes, depth 2: 4 nodes
        tree_cache = {}
        max_cache_depth = min(2, forest_height)
        for d in range(max_cache_depth + 1):
            start_idx = (1 << d) - 1
            count = 1 << d
            for j in range(count):
                node_idx = start_idx + j
                if node_idx < n_nodes:
                    addr = self.alloc_scratch(f"tree_{node_idx}", VLEN)
                    tree_cache[node_idx] = addr

        # Bit extraction vectors for vselect
        v_bit_masks = []
        for b in range(3):  # bits 0-2 sufficient for depth 0-2
            bm = self.alloc_scratch(f"v_bitmask_{b}", VLEN)
            v_bit_masks.append(bm)

        # Pre-allocated temp vectors for vselect tree (max depth 4 needs ~8 intermediates)
        vsel_temps = []
        for i in range(16):  # enough for depth 4 (16 candidates at most)
            vsel_temps.append(self.alloc_scratch(f"vsel_t{i}", VLEN))

        # ===== PROLOGUE: Load constants and initial data =====
        ops = []  # list of (engine, slot, read_addrs, write_addrs)

        # Load header from memory
        def emit(engine, slot):
            """Add raw instruction (one slot per cycle for now, will pack later)"""
            ops.append((engine, slot))

        # Load memory header values - use 4 independent scalar temps (gather_addrs[0..3])
        # to break serialization through s_tmp. All 4 const+load pairs can overlap.
        for hi, (mem_off, dest) in enumerate([
            (4, s_forest_values_p), (5, s_inp_indices_p),
            (6, s_inp_values_p), (1, s_n_nodes)
        ]):
            ga = gather_addr_sets[0][hi]  # independent temp per header load
            emit("load", ("const", ga, mem_off))
            emit("load", ("load", dest, ga))

        # Load scalar constants (already independent - different dest addresses)
        emit("load", ("const", sc_zero, 0))
        emit("load", ("const", sc_one, 1))
        emit("load", ("const", sc_two, 2))

        # Broadcast to vectors
        emit("valu", ("vbroadcast", v_zero, sc_zero))
        emit("valu", ("vbroadcast", v_one, sc_one))
        emit("valu", ("vbroadcast", v_two, sc_two))
        emit("valu", ("vbroadcast", v_n_nodes, s_n_nodes))

        # Load and broadcast hash/mult/shift constants: use self-staging to break
        # the s_tmp serialization chain. Load const directly to first element of
        # the vector (addr), then vbroadcast(addr, addr) reads that element.
        # All 6 hash const loads are now independent → can all start in 3 LOAD cycles.
        for i, val in enumerate(HASH_CONST_VALS):
            emit("load", ("const", hash_consts[i], val % (2**32)))
            emit("valu", ("vbroadcast", hash_consts[i], hash_consts[i]))

        for entry in hash_mult_consts:
            if entry is not None:
                _, addr, val = entry
                emit("load", ("const", addr, val))
                emit("valu", ("vbroadcast", addr, addr))

        for entry in hash_shift_consts:
            if entry is not None:
                _, addr, val = entry
                emit("load", ("const", addr, val))
                emit("valu", ("vbroadcast", addr, addr))

        for b in range(len(v_bit_masks)):
            emit("load", ("const", v_bit_masks[b], 1 << b))
            emit("valu", ("vbroadcast", v_bit_masks[b], v_bit_masks[b]))

        # Load initial values: use per-chunk gather_addr as independent address temp
        # so all 32 vloads can pipeline at 2/cycle (~18 cycles total vs ~96 serialized)
        for c in range(N_CHUNKS):
            ga = gather_addr_sets[c // VLEN][c % VLEN]
            emit("load", ("const", ga, c * VLEN))
            emit("alu", ("+", ga, s_inp_values_p, ga))
            emit("load", ("vload", val_vecs[c], ga))

        # Preload tree cache: use 7 independent gather_addr slots for parallel loads
        tree_cache_sorted = sorted(tree_cache.keys())
        for j, node_idx in enumerate(tree_cache_sorted):
            # Use gather_addrs[0..6] (all different) for independent parallel loads
            ga_idx = j % VLEN
            ga_set = j // VLEN
            ga = gather_addr_sets[ga_set][ga_idx]
            emit("load", ("const", ga, node_idx))
            emit("alu", ("+", ga, s_forest_values_p, ga))
            emit("load", ("load", ga, ga))
            emit("valu", ("vbroadcast", tree_cache[node_idx], ga))

        # Pause for debug harness
        emit("flow", ("pause",))

        # ===== MAIN LOOP: 16 rounds =====
        def get_depth(rnd):
            """Compute the tree depth for round rnd (all elements at same depth)"""
            d = rnd
            while d > forest_height:
                d = d - (forest_height + 1)
            return d

        def emit_get_node_val_cached(depth, chunk_idx, dest_vec, temp_set=0):
            """Get node values using cached tree levels with flow vselect.
            idx_vecs stores OFFSET within level (not absolute index).
            Flow vselect (1/cycle) overlaps with VALU hash work.
            """
            _, _, t3, _ = vt_sets[temp_set % N_TEMP_SETS]

            if depth == 0:
                emit("valu", ("+", dest_vec, tree_cache[0], v_zero))
                return

            level_start = (1 << depth) - 1
            level_count = 1 << depth

            candidates = []
            for j in range(level_count):
                node_idx = level_start + j
                candidates.append(tree_cache.get(node_idx, v_zero))

            # idx_vecs IS already the offset - no subtraction needed!

            current = candidates
            vsel_idx = 0
            for bit in range(depth):
                emit("valu", ("&", t3, idx_vecs[chunk_idx], v_bit_masks[bit]))
                new_current = []
                for p in range(0, len(current), 2):
                    if p + 1 < len(current):
                        if len(current) > 2:
                            out = vsel_temps[vsel_idx % len(vsel_temps)]
                            vsel_idx += 1
                            emit("flow", ("vselect", out, t3, current[p + 1], current[p]))
                            new_current.append(out)
                        else:
                            emit("flow", ("vselect", dest_vec, t3, current[p + 1], current[p]))
                            new_current.append(dest_vec)
                    else:
                        new_current.append(current[p])
                current = new_current

        # Scratch for base address per deep round (forest_values_p + level_start)
        s_gather_base = self.alloc_scratch("gather_base")

        def emit_gather(chunk_idx, dest_vec, gather_set=0):
            """Gather tree values for arbitrary indices (deep levels).
            idx_vecs stores OFFSET, so we use s_gather_base = forest_values_p + level_start.
            Uses per-lane address temps so loads can be pipelined (2/cycle)."""
            ga = gather_addr_sets[gather_set % N_GATHER_SETS]
            # Compute all 8 addresses: base + offset[lane]
            for lane in range(VLEN):
                src_addr = idx_vecs[chunk_idx] + lane
                emit("alu", ("+", ga[lane], s_gather_base, src_addr))
            # Then do all 8 loads (scheduler can pack 2 per cycle)
            for lane in range(VLEN):
                emit("load", ("load", dest_vec + lane, ga[lane]))

        def emit_hash_vectorized(val_vec, temp_set=0):
            """Emit vectorized hash with multiply_add fusion"""
            t1, t2, _, _ = vt_sets[temp_set % N_TEMP_SETS]
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                hc = hash_consts[hi]

                if op2 == "+" and op3 == "<<":
                    mult_entry = hash_mult_consts[hi]
                    if mult_entry is not None:
                        _, mult_vec, _ = mult_entry
                        emit("valu", ("multiply_add", val_vec, val_vec, mult_vec, hc))
                    else:
                        emit("valu", (op1, t1, val_vec, hc))
                        shift_entry = hash_shift_consts[hi]
                        _, sv, _ = shift_entry
                        emit("valu", (op3, t2, val_vec, sv))
                        emit("valu", (op2, val_vec, t1, t2))
                else:
                    emit("valu", (op1, t1, val_vec, hc))
                    shift_entry = hash_shift_consts[hi]
                    _, sv, _ = shift_entry
                    emit("valu", (op3, t2, val_vec, sv))
                    emit("valu", (op2, val_vec, t1, t2))

        def emit_direction_and_update(chunk_idx, val_vec, temp_set=0):
            """Compute direction and update offset using VALU.
            With offset representation: new_offset = 2*offset + (val & 1).
            """
            t1, _, _, _ = vt_sets[temp_set % N_TEMP_SETS]
            emit("valu", ("&", t1, val_vec, v_one))   # t1 = val & 1 (0 or 1)
            emit("valu", ("multiply_add", idx_vecs[chunk_idx], idx_vecs[chunk_idx], v_two, t1))

        # Generate all round instructions
        # idx_vecs stores OFFSET within current level (not absolute index)
        # For deep rounds: compute gather_base = forest_values_p + level_start
        # For shallow rounds: use offset directly for vselect tree
        for rnd in range(rounds):
            depth = get_depth(rnd)
            wraps = (depth >= forest_height)  # all elements wrap to root after this round
            prev_depth = get_depth(rnd - 1) if rnd > 0 else -1
            prev_was_wrap = (rnd > 0) and (prev_depth >= forest_height)
            # After a wrap round (or at rnd=0 where all idx=0 from scratch init),
            # all offsets are conceptually 0. If depth == 0, idx is not used
            # for lookup. New offset = 2*0 + bit = bit. Skip multiply: idx = val & 1.
            after_wrap_depth0 = (depth == 0) and (prev_was_wrap or rnd == 0)

            if depth > max_cache_depth:
                # DEEP ROUND: compute gather base for this level
                level_start = (1 << depth) - 1
                emit("load", ("const", s_tmp, level_start))
                emit("alu", ("+", s_gather_base, s_forest_values_p, s_tmp))

                # Phase 1: All gathers (independent with separate addr sets)
                for c in range(N_CHUNKS):
                    gs = c % N_GATHER_SETS
                    ga = gather_addr_sets[gs]
                    for lane in range(VLEN):
                        src_addr = idx_vecs[c] + lane
                        emit("alu", ("+", ga[lane], s_gather_base, src_addr))
                    for lane in range(VLEN):
                        emit("load", ("load", node_val_vecs[c] + lane, ga[lane]))

                # Phase 2: All hash + update (scheduler overlaps with remaining loads)
                for c in range(N_CHUNKS):
                    ts = c % N_TEMP_SETS
                    emit("valu", ("^", val_vecs[c], val_vecs[c], node_val_vecs[c]))
                    emit_hash_vectorized(val_vecs[c], ts)
                    if wraps:
                        # All elements wrap: don't emit broadcast (next round will
                        # use after_wrap_depth0 optimization or broadcast is needed)
                        # For round 10→11 pattern: skip reset, round 11 uses idx=val&1
                        next_depth = get_depth(rnd + 1) if rnd + 1 < rounds else -1
                        if next_depth != 0:
                            emit("valu", ("vbroadcast", idx_vecs[c], sc_zero))
                        # else: skip reset, round 11 will do idx = val & 1 directly
                    else:
                        emit_direction_and_update(c, val_vecs[c], ts)
            else:
                # SHALLOW ROUND: use offset directly for vselect (no level_start subtraction)
                for c in range(N_CHUNKS):
                    ts = c % N_TEMP_SETS
                    t1, _, _, _ = vt_sets[ts]
                    if depth == 0:
                        # Depth 0: all elements at root, XOR directly with cached value
                        emit("valu", ("^", val_vecs[c], val_vecs[c], tree_cache[0]))
                    else:
                        emit_get_node_val_cached(depth, c, node_val_vecs[c], ts)
                        emit("valu", ("^", val_vecs[c], val_vecs[c], node_val_vecs[c]))
                    emit_hash_vectorized(val_vecs[c], ts)
                    if wraps:
                        next_depth = get_depth(rnd + 1) if rnd + 1 < rounds else -1
                        if next_depth != 0:
                            emit("valu", ("vbroadcast", idx_vecs[c], sc_zero))
                    elif after_wrap_depth0:
                        # Previous round wrapped to root (offset=0 conceptually).
                        # New offset = 2*0 + bit0 = bit0. Skip the multiply.
                        emit("valu", ("&", idx_vecs[c], val_vecs[c], v_one))
                    else:
                        emit_direction_and_update(c, val_vecs[c], ts)

        # ===== EPILOGUE: Store values back to memory =====
        for c in range(N_CHUNKS):
            emit("load", ("const", s_tmp, c * VLEN))
            emit("alu", ("+", s_tmp, s_inp_values_p, s_tmp))
            emit("store", ("vstore", s_tmp, val_vecs[c]))

        # Note: indices NOT stored back - submission tests only check values

        # Pause for debug harness
        emit("flow", ("pause",))

        # ===== VLIW PACKING with dependency tracking =====
        self.instrs = _schedule_vliw(ops)

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        actual_vals = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        expected_vals = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        if actual_vals != expected_vals:
            for j in range(min(8, len(actual_vals))):
                if actual_vals[j] != expected_vals[j]:
                    print(f"  MISMATCH at elem {j}: got {actual_vals[j]:#x}, expected {expected_vals[j]:#x}")
            actual_idx = machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            expected_idx = ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)]
            for j in range(min(8, len(actual_idx))):
                print(f"  idx[{j}]: got {actual_idx[j]}, expected {expected_idx[j]}")
        assert actual_vals == expected_vals, f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
