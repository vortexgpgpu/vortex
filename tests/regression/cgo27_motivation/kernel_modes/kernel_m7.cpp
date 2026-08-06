// GPU-side program for mode 7 -- DTCU_socket, 4 engines, D -> that socket's L1.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m7.vxbin, which runs on
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

// mode 7: engine at socket scope, D lands in that socket's L1.
//
// One engine per socket and one descriptor per socket, so all of them run at once and
// each band's D lands in the L1 of the socket that computed it -- the placement this
// variant exists to model.
//
// The submitter is chosen by WHERE THE BLOCK LANDED, not by thread id: a core cannot
// address another socket's engine, so the mapping has to come from vx_core_id(). Socket
// size and count are build-time constants the kernel already carries.
__kernel void moti_dtcu_socket(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t core = (uint32_t)vx_core_id();
  if ((core % VX_CFG_SOCKET_SIZE) != 0)
    return;                                   // not this socket's submitter
  const uint32_t sock = core / VX_CFG_SOCKET_SIZE;
  const uint32_t nsock = VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE;

  const moti_slice sl = moti_slice_of(arg->M, nsock, sock);
  if (sl.nrow == 0)
    return;

  auto d = reinterpret_cast<dtensor_desc_t*>(arg->desc_addr) + sock;
  const uint64_t da = arg->desc_addr + (uint64_t)sock * sizeof(dtensor_desc_t);
  moti_fill_desc(d, arg, sl, MOTI_SOCKET_TILE_N);
  moti_publish_desc(da);
  while (0 == dtensor_socket_start(da))
    ;
  while (0 == dtensor_check(da))
    ;
}
#include "k_epilogue.h"
