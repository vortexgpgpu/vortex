#include "common.h"
#include <vx_spawn2.h>
#include <vx_dtensor.h>

// dtcu_basic: drive the disaggregated tensor core (DTCU). Launched as a single thread
// (1x1 grid, 1x1 block), so this body runs exactly once: it submits the GEMM descriptor
// and waits for the engine's completion flag. The DTCU runs the whole tiled GEMM
// autonomously and writes D; the SIMT core only submits and checks.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  // The engine is chosen by the INSTRUCTION, not by a descriptor field, so this has to
  // branch between two different inline-asm intrinsics -- there is nothing to
  // parameterise. A 0 ticket means the descriptor queue was full and nothing was
  // queued, so retry rather than silently skip the GEMM.
  if (arg->engine == DTCU_ENGINE_CLUSTER) {
    while (0 == dtensor_cluster_start(arg->desc_addr))
      ;
  } else {
    while (0 == dtensor_socket_start(arg->desc_addr))
      ;
  }
  while (0 == dtensor_check(arg->desc_addr))
    ;
}
