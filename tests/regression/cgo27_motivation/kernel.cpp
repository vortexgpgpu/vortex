// cgo27_motivation device code.
//
// Every HW path is a SEPARATE __kernel entry, selected host-side by name via
// vx_module_get_kernel (see main.cpp). All entries compute the SAME GEMM
// D = C + A*B (A row-major, B col-major, fp16 in / fp32 out) on the SAME input,
// so the only difference between modes is which compute/memory unit runs it.
//
//   moti_simt          mode 0    k_core.h  scalar MAC loop (sw fp16->fp32)
//   moti_tcu           mode 1    k_tcu.h   per-warp WMMA, global operand load
//   moti_tcu_dxa       mode 2    k_tcu.h   WMMA, DXA-staged smem (single-buffer)
//   moti_dtcu_cluster  mode 3    k_dtcu.h  descriptor engine at cluster scope (D->L2)
//   moti_dtcu_socket   mode 4    k_dtcu.h  descriptor engine at socket scope  (D->L1)
//   moti_tcu_pipe      mode 5    k_tcu.h   WMMA, register double-buffer
//   moti_tcu_dxa_pipe  mode 6    k_tcu.h   WMMA, DXA smem double-buffer
//
// This file is intentionally just the include list: the shared device helpers live
// in wmma_common.h and each unit's entries live in its own k_*.h, so a change to
// one execution path cannot disturb the others.

#include "wmma_common.h"   // ctx (tile geometry), h2f, wmma_seed_C / wmma_store_D
#include "k_core.h"        // mode 0
#include "k_tcu.h"         // modes 1, 2, 5, 6
#include "k_dtcu.h"        // modes 3, 4
