#ifndef _CGO27_K_DTCU_H_
#define _CGO27_K_DTCU_H_

// modes 7/8 — the two DTCU placement variants. The whole GEMM is one descriptor: the
// engine walks the tiles itself, so the kernel only fires the descriptor and waits.
// Launched 1x1x1 (a single thread), so no vx_thread_id guard is needed.
// Source lineage: dtcu_compare mode 1.
//
// TWO entries, not one entry branching on arg->mode, because the engine is selected by
// the start INSTRUCTION (RISCV_CUSTOM2 funct3) and an opcode cannot be picked at run
// time. The host selects the entry by name. The descriptor is identical for both.
//
// These are the only kernel entries that contain a dtensor start; keeping them isolated
// here is deliberate.

#include "wmma_common.h"
#include <vx_spawn2.h>
#include <vx_dtensor.h>

// mode 8: engine at cluster scope, D lands in L2.
__kernel void moti_dtcu_cluster(kernel_arg_t* __UNIFORM__ arg) {
  while (0 == dtensor_cluster_start(arg->desc_addr)) // 0 = queue full, nothing queued
    ;
  while (0 == dtensor_check(arg->desc_addr))
    ;
}

// mode 7: engine at socket scope, D lands in that socket's L1 dcache.
__kernel void moti_dtcu_socket(kernel_arg_t* __UNIFORM__ arg) {
  while (0 == dtensor_socket_start(arg->desc_addr))
    ;
  while (0 == dtensor_check(arg->desc_addr))
    ;
}

#endif // _CGO27_K_DTCU_H_
