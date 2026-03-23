#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Each thread exercises vx_packlb_f and vx_packlh_f over NUM_POINTS vectors.
// Layout of src (byte array):
//   For point p in thread t:  src[t*4*NUM_POINTS + p*4 + lane]  (stride = 1 byte)
//   For PACKLB: base = &src[t*4*NUM_POINTS + p*4], stride = 1
//     → result = b0 | (b1<<8) | (b2<<16) | (b3<<24)
//   For PACKLH: base = &src_u16[t*2*NUM_POINTS + p*2], stride = 2 bytes
//     → result = h0 | (h1<<16)
extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    auto src_ptr  = reinterpret_cast<const uint8_t*>(arg->src_addr);
    auto dst_lb   = reinterpret_cast<float*>(arg->dst_lb_addr);
    auto dst_lh   = reinterpret_cast<float*>(arg->dst_lh_addr);

    uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = 1; // byte stride between consecutive elements

    for (uint32_t p = 0; p < NUM_POINTS; ++p) {
        // 4 bytes at consecutive addresses → one packed float (PACKLB)
        const uint8_t* base_lb = src_ptr + (tid * 4 * NUM_POINTS + p * 4);
        dst_lb[tid * NUM_POINTS + p] = vx_packlb_f(base_lb, stride);

        // 2 halfwords at consecutive addresses → one packed float (PACKLH)
        const uint8_t* base_lh = src_ptr + (tid * 4 * NUM_POINTS + p * 4);
        dst_lh[tid * NUM_POINTS + p] = vx_packlh_f(base_lh, 2 /*halfword stride*/);
    }
}
