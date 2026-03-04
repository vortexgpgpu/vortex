#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  // DTCU only works on core 0
  // Issue the start command from the first thread of the first warp, and wait until completion
  if (vx_warp_id() == 0 && vx_thread_id() == 0) {
    vt::dtensor_start(arg->desc_addr);
    while (0 == vt::dtensor_poll()) {
      // busy wait
    }
  }
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  // 1 warp w/ grid=1x1, block=NUM_THREADS x 1
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
