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
#include <vx_intrinsics.h>

// mode 8: engine at cluster scope, D lands in L2. ONE descriptor for the whole GEMM --
// there is only one cluster engine, so slicing would just queue work behind itself.
// Launched 1x1x1, so no thread guard is needed.
__kernel void moti_dtcu_cluster(kernel_arg_t* __UNIFORM__ arg) {
  while (0 == dtensor_cluster_start(arg->desc_addr)) // 0 = queue full, nothing queued
    ;
  while (0 == dtensor_check(arg->desc_addr))
    ;
}

// mode 7: engine at socket scope, D lands in that socket's L1 dcache. There are
// NUM_SOCKETS of these engines, so the host splits the GEMM's ROWS into one slice per
// socket and this kernel submits slice s to socket s's own engine -- all of them run
// at once, and each slice's D lands in the L1 of the socket that computed it, which is
// the placement the variant exists to model.
//
// Launched one block per CORE, not per socket: a core cannot choose which engine it
// reaches (the instruction always goes to its own socket's), so the mapping has to come
// from where the block actually landed. One submitter per socket, the rest return.
__kernel void moti_dtcu_socket(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t ss = arg->socket_size ? arg->socket_size : 1u;
  const uint32_t core = (uint32_t)vx_core_id();
  if ((core % ss) != 0)
    return;                       // not this socket's submitter
  const uint32_t sock = core / ss;
  if (sock >= arg->num_slices)
    return;                       // more sockets than slices
  const uint64_t d = arg->desc_addr + (uint64_t)sock * sizeof(dtensor_desc_t);
  while (0 == dtensor_socket_start(d))
    ;
  while (0 == dtensor_check(d))
    ;
}

#endif // _CGO27_K_DTCU_H_
