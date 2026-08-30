#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Read the FULL 64-bit FP container produced by a packed load.
//
// The packed load and the fmv.x.d are fused into ONE asm block using an
// explicit FP register. As two separate statements the compiler is free to
// spill the float through a 32-bit fsw/flw pair, which would destroy (or
// silently recreate) the upper half -- exactly the bits under test. The
// instruction encoding is identical to vx_intrinsics.h's vx_packlb_f /
// vx_packlh_f: custom0, funct7=4, funct3=1/2.

//The inline functions are needed to prevent register spilling. 
// Register spilling silently erases the bug by doing proper NaN-boxing. 
// Uncomment the two lines to see all the tests pass.
__attribute__((always_inline))
inline uint64_t packlb_bits(const void* base, uint32_t stride) {
    uint64_t bits;
    __asm__ volatile (
        ".insn r %1, 1, 4, ft0, %2, %3\n\t"
        //"fsw  ft0, 12(sp)\n\t" /
        //"flw  ft0, 12(sp)\n\t"
        "fmv.x.d %0, ft0"
        : "=r"(bits) : "i"(RISCV_CUSTOM0), "r"(base), "r"(stride) : "ft0", "memory"
    );
    return bits;
}

__attribute__((always_inline))
inline uint64_t packlh_bits(const void* base, uint32_t stride) {
    uint64_t bits;
    __asm__ volatile (
        ".insn r %1, 2, 4, ft0, %2, %3\n\t"
        //"fsw  ft0, 12(sp)\n\t"
        //"flw  ft0, 12(sp)\n\t"
        "fmv.x.d %0, ft0"
        : "=r"(bits) : "i"(RISCV_CUSTOM0), "r"(base), "r"(stride) : "ft0", "memory"
    );
    return bits;
}

// Each thread exercises vx_packlb_f and vx_packlh_f over NUM_POINTS vectors.
// Layout of src (byte array):
//   For point p in thread t:  src[t*4*NUM_POINTS + p*4 + lane]  (stride = 1 byte)
//   For PACKLB: base = &src[t*4*NUM_POINTS + p*4], stride = 1
//     → result = b0 | (b1<<8) | (b2<<16) | (b3<<24)
//   For PACKLH: base = &src_u16[t*2*NUM_POINTS + p*2], stride = 2 bytes
//     → result = h0 | (h1<<16)
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    auto src_ptr  = reinterpret_cast<const uint8_t*>(arg->src_addr);
    // uint64_t, not float: a float store would drop the NaN-box half
    auto dst_lb   = reinterpret_cast<uint64_t*>(arg->dst_lb_addr);
    auto dst_lh   = reinterpret_cast<uint64_t*>(arg->dst_lh_addr);

    uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = 1; // byte stride between consecutive elements

    for (uint32_t p = 0; p < NUM_POINTS; ++p) {
        // 4 bytes at consecutive addresses → one packed float (PACKLB)
        const uint8_t* base_lb = src_ptr + (tid * 4 * NUM_POINTS + p * 4);
        dst_lb[tid * NUM_POINTS + p] = packlb_bits(base_lb, stride);

        // 2 halfwords at consecutive addresses → one packed float (PACKLH)
        const uint8_t* base_lh = src_ptr + (tid * 4 * NUM_POINTS + p * 4);
        dst_lh[tid * NUM_POINTS + p] = packlh_bits(base_lh, 2 /*halfword stride*/);
    }
}
