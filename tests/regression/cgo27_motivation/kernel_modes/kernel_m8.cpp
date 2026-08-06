// GPU-side program for mode 8 -- DTCU_cluster, 1 engine, D -> L2.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m8.vxbin, which runs on
// the GPU. It has no main(); its entry point is the __kernel below. The host program that
// opens the device, uploads A/B/C, launches this and reads D back is main.cpp, built for
// x86 -- that is where main() lives.
//
// It contains this mode's kernel and NOTHING else, which is the point. In the old
// all-in-one kernel.vxbin every mode occupied address space even though only one ran, and
// address decides icache set: adding modes 3/4 moved mode 2 from 15,468 to 24,106 cycles
// with a byte-identical kernel body. Here every mode's code starts at 0x180000034
// whatever else is in the tree.
//
// k_epilogue.h too: the engine is GEMM-only, so an elementwise app costs it a
// SECOND launch over D. That is the cost asymmetry the app sweep measures.

#include "wmma_common.h"
#include "k_dtcu_desc.h"
#include <vx_spawn2.h>

// mode 8: engine at cluster scope, D lands in L2.
//
// One engine, but NUM_CORES descriptors: the GEMM is split per core and each core
// submits its own band. That does not add compute parallelism -- there is still one MAC
// array -- so what it measures is whether the engine can absorb a queue of small
// descriptors as cheaply as one large one, i.e. the per-descriptor cost of DESC_REQ /
// DESC_WAIT and pipeline refill. Queue depth is 2 x NUM_CORES, so all of them fit.
__kernel void moti_dtcu_cluster(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t core = (uint32_t)vx_core_id();
  const moti_slice sl = moti_slice_of(arg->M, VX_CFG_NUM_CORES, core);
  if (sl.nrow == 0)
    return;                                   // fewer rows than cores; nothing to submit

  auto d = reinterpret_cast<dtensor_desc_t*>(arg->desc_addr) + core;
  const uint64_t da = arg->desc_addr + (uint64_t)core * sizeof(dtensor_desc_t);
  moti_fill_desc(d, arg, sl, MOTI_CLUSTER_TILE_N);
  moti_publish_desc(da);
  while (0 == dtensor_cluster_start(da))      // 0 = queue full, nothing was queued
    ;
  while (0 == dtensor_check(da))
    ;
}
#include "k_epilogue.h"
