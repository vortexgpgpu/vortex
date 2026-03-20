#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    // Virgo-style synchronization
    // Still assumes that core 0 can only issue comand
    auto num_cores = vx_num_cores();
    bool is_leader = (vx_core_id() == 0) && (vx_warp_id() == 0) && (vx_thread_id() == 0);

    // memory fence
    vx_fence();

	// global barrier
	vx_barrier(0x80000000, num_cores);

    if (is_leader) {
        vt::dtensor_start(arg->desc_addr);
        while (0 == vt::dtensor_poll()) {
            // busy wait
        }
    }

    // Commit before moving on
    vx_fence();
    vx_barrier(0x80000000, num_cores);
}

int main() {
    auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
    // 1 warp w/ grid=1x1, block=NUM_THREADS x 1
    return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
