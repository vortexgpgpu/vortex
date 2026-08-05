#ifndef _CGO27_K_DTCU_H_
#define _CGO27_K_DTCU_H_

// modes 3/4 — DTCU (cluster-level disaggregated tensor core). The whole GEMM is
// one descriptor: the engine walks the tiles itself, so the kernel only fires the
// descriptor and waits. Launched 1x1x1 (a single thread), so no vx_thread_id
// guard is needed. Modes 3 and 4 run the SAME kernel — they differ only in the
// host-built descriptor's DTENSOR_FLAG_NO_TMA bit (see desc.h / main.cpp).
// Source lineage: dtcu_compare mode 1.
//
// This is the only kernel entry that contains dtensor_start; keeping it isolated
// here is deliberate.

#include "wmma_common.h"
#include <vx_spawn2.h>
#include <vx_dtensor.h>

__kernel void moti_dtcu(kernel_arg_t* __UNIFORM__ arg) {
  while (0 == dtensor_cluster_start(arg->desc_addr)) // 0 = queue full, nothing queued
    ;
  while (0 == dtensor_check(arg->desc_addr))
    ;
}

#endif // _CGO27_K_DTCU_H_
