# SST Phase 3 integration test for vortex.VortexGPGPU.
#
# Wires the VortexGPGPU component's optional `memIface` SubComponent slot
# through an L1 cache to a memHierarchy.MemController. Every memory request
# accepted by Vortex's local DRAM model is mirrored to the SST memHierarchy
# as a StandardMem::Read or Write event, so memHierarchy can model timing /
# capacity / contention alongside Vortex's own simulation.
#
# This is the Phase 3 demonstrator from docs/proposals/sst_simx_v3_proposal.md.
# The local data path stays in Vortex (RAM is authoritative); SST sees
# every transaction but doesn't have to serve data back. That gives us
# meaningful integration without forcing v3's TLM data path through SST.

import sst

# --- Vortex GPGPU component (single-warp hello kernel) -----------------------
gpu = sst.Component("gpu0", "vortex.VortexGPGPU")
gpu.addParams({
    "clock":   "1GHz",
    "program": "tests/kernel/hello/hello.vxbin",
})

# Vortex's StandardMem-side adapter
gpu_mem_iface = gpu.setSubComponent("memIface", "memHierarchy.standardInterface")

# --- L1 cache between Vortex and memory --------------------------------------
# A cache is required because memHierarchy.MemController routes via MemLink
# and only registers its address range when there's an upstream cache that
# advertises destinations.
l1 = sst.Component("l1cache", "memHierarchy.Cache")
l1.addParams({
    "access_latency_cycles": "2",
    "cache_frequency":       "1GHz",
    "replacement_policy":    "lru",
    "coherence_protocol":    "MESI",
    "associativity":         "4",
    "cache_line_size":       "64",
    "L1":                    "1",
    "cache_size":            "8KiB",
})

# --- Memory controller + simple backend (host RAM-backed) --------------------
memctrl = sst.Component("memctrl0", "memHierarchy.MemController")
memctrl.addParams({
    "clock":          "1GHz",
    "addr_range_end": 0x100000000 - 1,  # 4 GB
})
memory = memctrl.setSubComponent("backend", "memHierarchy.simpleMem")
memory.addParams({
    "access_time": "10ns",
    "mem_size":    "4GiB",
})

# --- Wiring ------------------------------------------------------------------
# Vortex GPGPU → L1 cache
link_gpu_l1 = sst.Link("link_gpu_l1")
link_gpu_l1.connect((gpu_mem_iface, "lowlink", "1ns"),
                    (l1,            "highlink", "1ns"))

# L1 cache → MemController
link_l1_mem = sst.Link("link_l1_mem")
link_l1_mem.connect((l1,      "lowlink",  "1ns"),
                    (memctrl, "highlink", "1ns"))
