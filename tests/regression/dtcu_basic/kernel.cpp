#include "common.h"
#include <vx_spawn2.h>
#include <vx_dtensor.h>

// dtcu_basic: drive the disaggregated tensor core (DTCU). Launched as a single thread
// (1x1 grid, 1x1 block), so this body runs exactly once: it fires the GEMM descriptor
// and spins on the done bit. The DTCU engine runs the whole tiled GEMM autonomously
// and writes D; the SIMT core only issues start/poll.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  dtensor_start(arg->desc_addr);
  while (0 == dtensor_poll()) {
    // busy-wait until the DTCU signals completion
  }
}
