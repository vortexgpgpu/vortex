// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The intrinsics implemented use RISC-V assembler pseudo-directives defined here:
// https://sourceware.org/binutils/docs/as/RISC_002dV_002dFormats.html

#ifndef __VX_INTRINSICS_H__
#define __VX_INTRINSICS_H__

#include <stddef.h>
#include <VX_config.h>
#include <VX_types.h>

#if defined(__clang__)
#define __UNIFORM__   __attribute__((annotate("vortex.uniform")))
#define __DIVERGENT__ __attribute__((annotate("vortex.divergent")))
#else
#define __UNIFORM__
#define __DIVERGENT__
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __ASSEMBLY__
#define __ASM_STR(x)	x
#else
#define __ASM_STR(x)	#x
#endif

__asm__ (".set reg_x0  ,  0");
__asm__ (".set reg_x1  ,  1");
__asm__ (".set reg_x2  ,  2");
__asm__ (".set reg_x3  ,  3");
__asm__ (".set reg_x4  ,  4");
__asm__ (".set reg_x5  ,  5");
__asm__ (".set reg_x6  ,  6");
__asm__ (".set reg_x7  ,  7");
__asm__ (".set reg_x8  ,  8");
__asm__ (".set reg_x9  ,  9");
__asm__ (".set reg_x10 , 10");
__asm__ (".set reg_x11 , 11");
__asm__ (".set reg_x12 , 12");
__asm__ (".set reg_x13 , 13");
__asm__ (".set reg_x14 , 14");
__asm__ (".set reg_x15 , 15");
__asm__ (".set reg_x16 , 16");
__asm__ (".set reg_x17 , 17");
__asm__ (".set reg_x18 , 18");
__asm__ (".set reg_x19 , 19");
__asm__ (".set reg_x20 , 20");
__asm__ (".set reg_x21 , 21");
__asm__ (".set reg_x22 , 22");
__asm__ (".set reg_x23 , 23");
__asm__ (".set reg_x24 , 24");
__asm__ (".set reg_x25 , 25");
__asm__ (".set reg_x26 , 26");
__asm__ (".set reg_x27 , 27");
__asm__ (".set reg_x28 , 28");
__asm__ (".set reg_x29 , 29");
__asm__ (".set reg_x30 , 30");
__asm__ (".set reg_x31 , 31");

__asm__ (".set reg_zero,  0");
__asm__ (".set reg_ra  ,  1");
__asm__ (".set reg_sp  ,  2");
__asm__ (".set reg_gp  ,  3");
__asm__ (".set reg_tp  ,  4");
__asm__ (".set reg_t0  ,  5");
__asm__ (".set reg_t1  ,  6");
__asm__ (".set reg_t2  ,  7");
__asm__ (".set reg_s0  ,  8");
__asm__ (".set reg_s1  ,  9");
__asm__ (".set reg_a0  , 10");
__asm__ (".set reg_a1  , 11");
__asm__ (".set reg_a2  , 12");
__asm__ (".set reg_a3  , 13");
__asm__ (".set reg_a4  , 14");
__asm__ (".set reg_a5  , 15");
__asm__ (".set reg_a6  , 16");
__asm__ (".set reg_a7  , 17");
__asm__ (".set reg_s2  , 18");
__asm__ (".set reg_s3  , 19");
__asm__ (".set reg_s4  , 20");
__asm__ (".set reg_s5  , 21");
__asm__ (".set reg_s6  , 22");
__asm__ (".set reg_s7  , 23");
__asm__ (".set reg_s8  , 24");
__asm__ (".set reg_s9  , 25");
__asm__ (".set reg_s10 , 26");
__asm__ (".set reg_s11 , 27");
__asm__ (".set reg_t3  , 28");
__asm__ (".set reg_t4  , 29");
__asm__ (".set reg_t5  , 30");
__asm__ (".set reg_t6  , 31");

#define RISCV_CUSTOM0   0x0B
#define RISCV_CUSTOM1   0x2B
#define RISCV_CUSTOM2   0x5B
#define RISCV_CUSTOM3   0x7B

#define RISCV_INSN_R(opcode, func3, func7, rs1, rs2) ({         \
    size_t __r;                                                 \
    __asm__ __volatile__ (                                      \
        ".word ((" __ASM_STR(opcode) ") | (reg_%0 << 7) | (reg_%1 << 15) | (reg_%2 << 20) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func7) ") << 25));" \
        : "=r" (__r): "r" (rs1), "r" (rs2                       \
    );                                                          \
    __r;                                                        \
})

#define RISCV_INSN_R_000(opcode, func3, func7, imm1, imm2) ({   \
    __asm__ __volatile__ (                                      \
        ".word ((" __ASM_STR(opcode) ") | (" __ASM_STR(imm1) " << 15) | (" __ASM_STR(imm2) " << 20) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func7) ") << 25));" \
        ::                                                      \
    );                                                          \
})

#define RISCV_INSN_R_010(opcode, func3, func7, rs1, imm2)  ({   \
    __asm__ __volatile__ (                                      \
        ".word ((" __ASM_STR(opcode) ") | (reg_%0 << 15) | (" __ASM_STR(imm2) " << 20) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func7) ") << 25));" \
        :: "r" (rs1)                                            \
    );                                                          \
})

#define RISCV_INSN_R_011(opcode, func3, func7, rs1, rs2) ({     \
    __asm__ __volatile__ (                                      \
        ".word ((" __ASM_STR(opcode) ") | (reg_%0 << 15) | (reg_%1 << 20) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func7) ") << 25));" \
        :: "r" (rs1), "r" (rs2)                                 \
    );                                                          \
})

#define RISCV_INSN_R_100(opcode, func3, func7, imm1, imm2) ({   \
    size_t __r;                                                 \
    __asm__ __volatile__ (                                      \
        ".word ((" __ASM_STR(opcode) ") | (reg_%0 << 7) | (" __ASM_STR(imm1) " << 15) | (" __ASM_STR(imm2) " << 20) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func7) ") << 25));" \
        : "=r" (__r) :                                          \
    );                                                          \
    __r;                                                        \
})

#define RISCV_INSN_R4(opcode, func3, func2, rs1, rs2, rs3) ({   \
    size_t __r;                                                 \
    __asm__ __volatile__ (                                      \
        ".word ((" __ASM_STR(opcode) ") | (reg_%0 << 7) | (reg_%1 << 15) | (reg_%2 << 20) | (reg_%3 << 27) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func2) ") << 25));" \
        : "=r" (__r): "r" (rs1), "r" (rs2), "r" (rs3)           \
    );                                                          \
    __r;                                                        \
})

#define RISCV_INSN_R4_0111(opcode, func3, func2, rs1, rs2, rs3) ({  \
    __asm__ __volatile__ (                                          \
        ".word ((" __ASM_STR(opcode) ") | (reg_%0 << 15) | (reg_%1 << 20) | (reg_%2 << 27) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func2) ") << 25));" \
        :: "r" (rs1), "r" (rs2), "r" (rs3)                          \
    );                                                              \
})

#define csr_read(csr) ({                        \
	size_t __r;	               		            \
	__asm__ __volatile__ ("csrr %0, %1" : "=r" (__r) : "i" (csr) : "memory"); \
	__r;							            \
})

#define csr_write(csr, val)	({                  \
	size_t __v = (size_t)(val);                 \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ __volatile__ ("csrw %0, %1" :: "i" (csr), "i" (__v) : "memory");  \
    else                                        \
        __asm__ __volatile__ ("csrw %0, %1"	:: "i" (csr), "r" (__v) : "memory");  \
})

#define csr_swap(csr, val) ({                   \
    size_t __r;                                 \
	size_t __v = (size_t)(val);	                \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ __volatile__ ("csrrw %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v) : "memory"); \
    else                                        \
        __asm__ __volatile__ ("csrrw %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v) : "memory"); \
	__r;						                \
})

#define csr_read_set(csr, val) ({               \
	size_t __r;                                 \
	size_t __v = (size_t)(val);	                \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ __volatile__ ("csrrs %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v) : "memory"); \
    else                                        \
        __asm__ __volatile__ ("csrrs %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v) : "memory"); \
	__r;							            \
})

#define csr_set(csr, val) ({                    \
	size_t __v = (size_t)(val);	                \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ __volatile__ ("csrs %0, %1"	:: "i" (csr), "i" (__v) : "memory");  \
    else                                        \
        __asm__ __volatile__ ("csrs %0, %1"	:: "i" (csr), "r" (__v) : "memory");  \
})

#define csr_read_clear(csr, val) ({             \
	size_t __r;                                 \
	size_t __v = (size_t)(val);	                \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ __volatile__ ("csrrc %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v) : "memory"); \
    else                                        \
        __asm__ __volatile__ ("csrrc %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v) : "memory"); \
	__r;							            \
})

#define csr_clear(csr, val)	({                  \
	size_t __v = (size_t)(val);                 \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ __volatile__ ("csrc %0, %1" :: "i" (csr), "i" (__v) : "memory"); \
    else                                        \
        __asm__ __volatile__ ("csrc %0, %1"	:: "i" (csr), "r" (__v) : "memory"); \
})

//TODO: Make these functions use the new RISCV defines (such as RISCV_INSN_R4)
// Set thread mask
inline void vx_tmc(size_t thread_mask) {
    asm volatile (".insn r %0, 0, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(thread_mask));
}

// disable all threads in the current warp
inline void vx_tmc_zero() {
    asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM0));
}

// switch execution to single thread zero
inline void vx_tmc_one() {
    asm volatile (
        "li a0, 1\n\t"  // Load immediate value 1 into a0 (x10) register
        ".insn r %0, 0, 0, x0, a0, x0" :: "i"(RISCV_CUSTOM0) : "a0"
    );
}

// Set thread predicate
inline void vx_pred(size_t condition, size_t thread_mask) {
    asm volatile (".insn r %0, 5, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask));
}

// Set thread not predicate
inline void vx_pred_n(int condition, int thread_mask) {
    asm volatile (".insn r %0, 5, 0, x1, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask));
}

// Spawn warps
typedef void (*vx_wspawn_pfn)();
inline void vx_wspawn(size_t num_warps, vx_wspawn_pfn func_ptr) {
    asm volatile (".insn r %0, 1, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(num_warps), "r"(func_ptr));
}

// Split on a predicate
inline size_t vx_split(size_t predicate) {
    size_t ret;
    asm volatile (".insn r %1, 2, 0, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Split on a not predicate
inline size_t vx_split_n(size_t predicate) {
    size_t ret;
    asm volatile (".insn r %1, 2, 0, %0, %2, x1" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Join
inline void vx_join(size_t stack_ptr) {
    asm volatile (".insn r %0, 3, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(stack_ptr));
}

// Warp Barrier
inline void vx_barrier(size_t barried_id, size_t num_warps) {
    asm volatile (".insn r %0, 4, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(barried_id), "r"(num_warps));
}

// Return current thread identifier
inline int vx_thread_id() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_THREAD_ID));
    return ret;
}

// Return current warp identifier
inline int vx_warp_id() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_WARP_ID));
    return ret;
}

// Return current core identifier
inline int vx_core_id() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_CORE_ID));
    return ret;
}

// Return active threads mask
inline int vx_active_threads() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_ACTIVE_THREADS));
    return ret;
}

// Return active warps mask
inline int vx_active_warps() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_ACTIVE_WARPS));
    return ret;
}

// Return the number of threads per warp
inline int vx_num_threads() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_THREADS));
    return ret;
}

// Return the number of warps per core
inline int vx_num_warps() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_WARPS));
    return ret;
}

// Return the number of cores per cluster
inline int vx_num_cores() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_CORES));
    return ret;
}

// Return the number of barriers
inline int vx_num_barriers() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_BARRIERS));
    return ret;
}

// Return the hart identifier (thread id accross the processor)
inline int vx_hart_id() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_MHARTID));
    return ret;
}

inline void vx_fence() {
    asm volatile ("fence iorw, iorw");
}

// Texture load
#define vx_tex(stage, u, v, lod) \
    RISCV_INSN_R4(RISCV_CUSTOM1, 0, stage, u, v, lod)

// Conditional move
inline size_t vx_cmov(size_t c, size_t t, size_t f) {
	return RISCV_INSN_R4(RISCV_CUSTOM1, 1, 0, c, t, f);
}

// Rop write
inline void vx_rop(size_t x, size_t y, size_t face, size_t color, size_t depth) {
    size_t pos_face = (y << 16) | (x << 1) | face;
    RISCV_INSN_R4_0111(RISCV_CUSTOM1, 1, 1, pos_face, color, depth);
}

// Integer multiply add
#define vx_imadd(a, b, c, shift) \
    RISCV_INSN_R4(RISCV_CUSTOM1, 2, shift, a, b, c)

// Raster load
inline size_t vx_rast() {
    return RISCV_INSN_R_100(RISCV_CUSTOM0, 0, 1, 0, 0);
}

inline void vx_vec_vvaddint32(size_t n, const int *a, const int *b, int *c) {

    //set the vector length for the maximum number of elements [VLEN 256-bit/32-bit = 8 elements]
    //for all iterations until n < 8

    //vsetvli need to be reset for the first time
    asm volatile ("vsetvli t0, %[n], e32" : : [n] "r" (n));

    while ( n > 8) {
        asm volatile (
            "vsetvli t0, %[n], e32\n\t"

            //load a and b
            "vle32.v v0, (%[a])\n\t"
            "vle32.v v1, (%[b])\n\t"

            //add a, b and store in c
            "vadd.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"

            // Decrement n and increment pointers of a, b, c to next 8 elements
            "sub %[n], %[n], t0\n\t"
            "slli t1, t0, 2\n\t"

            "add %[a], %[a], t1\n\t"
            "add %[b], %[b], t1\n\t"
            "add %[c], %[c], t1\n\t"

            : [n]"+r"(n), [a]"+r"(a), [b]"+r"(b), [c]"+r"(c)
            :
            : "t0", "t1", "v0", "v1", "v2"
        );
    }

    // last iteration range [0 < n < 8]
    if (n > 0) {
        asm volatile (
            //set the vector length to the remaining elements and mask the rest
            "vsetvli t0, %[n], e32, m1\n\t"

            //load a, b
            "vle32.v v0, (%[a])\n\t"
            "vle32.v v1, (%[b])\n\t"

            //add a, b and store in c
            "vadd.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"

            : [n]"+r"(n), [a]"+r"(a), [b]"+r"(b), [c]"+r"(c)
            :
            : "t0", "v0", "v1", "v2"
        );
    }
}


inline void vx_vec_vvmulint32(size_t n, const int *a, const int *b, int *c) {

    asm volatile ("vsetvli t0, %[n], e32" : : [n] "r" (n));

    while (n > 8){
        asm volatile (

            "vsetvli t0, %[n], e32\n\t"

            //load a, b
            "vle32.v v0, (%[a])\n\t"
            "vle32.v v1, (%[b])\n\t"

            //multiply a, b and store in c
            "vmul.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"

            // Decrement n and increment pointers of a, b, c to next 8 elements
            "sub %[n], %[n], t0\n\t"
            "slli t1, t0, 2\n\t"

            "add %[a], %[a], t1\n\t"
            "add %[b], %[b], t1\n\t"
            "add %[c], %[c], t1\n\t"

            : [n]"+r"(n), [a]"+r"(a), [b]"+r"(b), [c]"+r"(c)
            :
            : "t0", "t1", "v0", "v1", "v2"
        );
    }

    // last iteration range [0 < n < 8]
    if (n > 0) {
        asm volatile (
            //set the vector length to the remaining elements and mask the vector [m8]
            "vsetvli t0, %[n], e32, m1\n\t"
            "vle32.v v0, (%[a])\n\t"
            "vle32.v v1, (%[b])\n\t"
            "vmul.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"
            // No need to update n and pointers here because we don't use them anymore
            : [n]"+r"(n), [a]"+r"(a), [b]"+r"(b), [c]"+r"(c)
            :
            : "t0", "v0", "v1", "v2"
        );
    }

}


inline void vx_vec_vvaddfloat32(size_t n, const float *a, const float *b, float *c) {

    asm volatile ("vsetvli t0, %[n], e32\n\t" : [n]"+r"(n));

    while (n > 8){
        asm volatile (
            "vsetvli t0, %[n], e32\n\t"

            //load a, b
            "vle32.v v0, (%[a])\n\t"
            "vle32.v v1, (%[b])\n\t"

            //add a, b and store in c
            "vfadd.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"

            // Decrement n and increment pointers of a, b, c to next 8 elements
            "sub %[n], %[n], t0\n\t"
            "slli t1, t0, 2\n\t"

            "add %[a], %[a], t1\n\t"
            "add %[b], %[b], t1\n\t"
            "add %[c], %[c], t1\n\t"

            : [n]"+r"(n), [a]"+r"(a), [b]"+r"(b), [c]"+r"(c)
            :
            : "t0", "t1", "v0", "v1", "v2"
        );

    }

    
    // last iteration range [0 < n < 8]
    if (n > 0) {
        asm volatile (
            //set the vector length to the remaining elements and mask the rest
            "vsetvli t0, %[n], e32, m1\n\t"

            "vle32.v v0, (%[a])\n\t"
            "vle32.v v1, (%[b])\n\t"

            "vfadd.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"

            // No need to update n and pointers here because we don't use them anymore
            : [n]"+r"(n), [a]"+r"(a), [b]"+r"(b), [c]"+r"(c)
            :
            : "t0", "v0", "v1", "v2"
        );
    }

}





inline void vx_vec_scalar_add_f32(size_t n, float scalar, const float *a, float *c) {

    asm volatile ("vsetvli t0, %[n], e32" : : [n] "r" (n));

    while(n > 8){
        asm volatile(
            // Load a
            "vle32.v v0, (%[a])\n\t"

            // Load scalar into vector register v1
            "vfmv.v.f v1, %[scalar]\n\t"

            // Add a, scalar and store in c
            "vfadd.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"

            // Decrement n and increment pointers of a, c to next 8 elements
            "sub %[n], %[n], t0\n\t"
            "slli t1, t0, 2\n\t"

            "add %[a], %[a], t1\n\t"
            "add %[c], %[c], t1\n\t"

            : [n]"+r"(n), [a]"+r"(a), [c]"+r"(c)
            : [scalar]"f"(scalar)
            : "t0", "t1", "v0", "v1", "v2"
        );
    }

    // last iteration range [0 < n < 8]
    if (n > 0) {
        asm volatile (
            //set the vector length to the remaining elements and mask the vector [m8]
            "vsetvli t0, %[n], e32, m1\n\t"
            "vle32.v v0, (%[a])\n\t"

            // Load scalar into vector register v1
            "vfmv.v.f v1, %[scalar]\n\t"

            // Add a, scalar and store in c
            "vfadd.vv v2, v0, v1\n\t"
            "vse32.v v2, (%[c])\n\t"
            
            // No need to update n and pointers here because we don't use them anymore
            : [n]"+r"(n), [a]"+r"(a), [c]"+r"(c)
            : [scalar]"f"(scalar)
            : "t0", "v0", "v1", "v2"
        );
    }
}

#define __if(b) vx_split(b); \
                if (b) 

#define __else else

#define __endif vx_join();

#ifdef __cplusplus
}
#endif

#endif // __VX_INTRINSICS_H__
