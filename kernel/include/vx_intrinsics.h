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
#include <stdint.h>
#include <VX_types.h>

#if defined(__clang__)
#define __UNIFORM__   __attribute__((annotate("vortex.uniform")))
#else
#define __UNIFORM__
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RISCV_CUSTOM0   0x0B
#define RISCV_CUSTOM1   0x2B
#define RISCV_CUSTOM2   0x5B
#define RISCV_CUSTOM3   0x7B

#define RISCV_INSN_R(opcode7, funct3, funct7, rd, rs1, rs2) ( \
    ((funct7 & 0x7F) << 25) | \
    ((rs2 & 0x1F) << 20) | \
    ((rs1 & 0x1F) << 15) | \
    ((funct3 & 0x7) << 12) | \
    ((rd & 0x1F) << 7) | \
    (opcode7 & 0x7F) \
)

#define RISCV_INSN_R4(opcode7, funct3, funct2, rd, rs1, rs2, rs3) ( \
    ((rs3 & 0x1F) << 27) | \
    ((funct2 & 0x3) << 25) | \
    ((rs2 & 0x1F) << 20) | \
    ((rs1 & 0x1F) << 15) | \
    ((funct3 & 0x7) << 12) | \
    ((rd & 0x1F) << 7) | \
    (opcode7 & 0x7F) \
)

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

// Set thread mask
inline void vx_tmc(int thread_mask) {
    __asm__ volatile (".insn r %0, 0, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(thread_mask));
}

// disable all threads in the current warp
inline void vx_tmc_zero() {
    __asm__ volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM0));
}

// switch execution to single thread zero
inline void vx_tmc_one() {
    __asm__ volatile (
        "li a0, 1\n\t"  // Load immediate value 1 into a0 (x10) register
        ".insn r %0, 0, 0, x0, a0, x0" :: "i"(RISCV_CUSTOM0) : "a0"
    );
}

// Set thread predicate
inline void vx_pred(int condition, int thread_mask) {
    __asm__ volatile (".insn r %0, 5, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask));
}

// Set thread not predicate
inline void vx_pred_n(int condition, int thread_mask) {
    __asm__ volatile (".insn r %0, 5, 0, x1, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask));
}

// Spawn warps
typedef void (*vx_wspawn_pfn)();
inline void vx_wspawn(int num_warps, vx_wspawn_pfn func_ptr) {
    __asm__ volatile (".insn r %0, 1, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(num_warps), "r"(func_ptr));
}

// Split on a predicate
inline int vx_split(int predicate) {
    int ret;
    __asm__ volatile (".insn r %1, 2, 0, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Split on a not predicate
inline int vx_split_n(int predicate) {
    int ret;
    __asm__ volatile (".insn r %1, 2, 0, %0, %2, x1" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Join
inline void vx_join(int stack_ptr) {
    __asm__ volatile (".insn r %0, 3, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(stack_ptr));
}

// Warp Barrier
inline void vx_barrier(int barried_id, int num_warps) {
    __asm__ volatile (".insn r %0, 4, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(barried_id), "r"(num_warps));
}

// Return current thread identifier
inline int vx_thread_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_THREAD_ID));
    return ret;
}

// Return current warp identifier
inline int vx_warp_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_WARP_ID));
    return ret;
}

// Return current core identifier
inline int vx_core_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_CORE_ID));
    return ret;
}

// Return active threads mask
inline int vx_active_threads() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_ACTIVE_THREADS));
    return ret;
}

// Return active warps mask
inline int vx_active_warps() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_ACTIVE_WARPS));
    return ret;
}

// Return the number of threads per warp
inline int vx_num_threads() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_THREADS));
    return ret;
}

// Return the number of warps per core
inline int vx_num_warps() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_WARPS));
    return ret;
}

// Return the number of cores per cluster
inline int vx_num_cores() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_CORES));
    return ret;
}

// Return the hart identifier (thread id accross the processor)
inline int vx_hart_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_MHARTID));
    return ret;
}

inline void vx_fence() {
    __asm__ volatile ("fence iorw, iorw");
}

typedef float mf32x8_t __attribute__((vector_size(8*4))); // 8 x f32 registers

#define MAKE_VX_WSETM_F32(f0, f1, f2, f3, f4, f5, f6, f7) \
    mf32x8_t ret; \
    register float fd0 __asm__(f0); \
    register float fd1 __asm__(f1); \
    register float fd2 __asm__(f2); \
    register float fd3 __asm__(f3); \
    register float fd4 __asm__(f4); \
    register float fd5 __asm__(f5); \
    register float fd6 __asm__(f6); \
    register float fd7 __asm__(f7); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd0): "r"(value)); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd1): "r"(value)); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd2): "r"(value)); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd3): "r"(value)); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd4): "r"(value)); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd5): "r"(value)); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd6): "r"(value)); \
    __asm__ volatile("fmv.w.x %0, %1" : "=f"(fd7): "r"(value)); \
    ret = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7}; \
    return ret

__attribute__((always_inline)) mf32x8_t vx_wsetm_a_f32(size_t value) {
    MAKE_VX_WSETM_F32("f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15");
}

__attribute__((always_inline)) mf32x8_t vx_wsetm_b_f32(size_t value) {
    MAKE_VX_WSETM_F32("f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
}

__attribute__((always_inline)) mf32x8_t vx_wsetm_c_f32(size_t value) {
    MAKE_VX_WSETM_F32("f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7");
}

__attribute__((always_inline)) mf32x8_t vx_wsetm_d_f32(size_t value) {
    MAKE_VX_WSETM_F32("f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23");
}

#define MAKE_VX_WLDM_D_F32(f0, f1, f2, f3, f4, f5, f6, f7) \
    mf32x8_t ret; \
    auto base = (const float*)src + row * ldm; \
    register float fd0 __asm__(f0); \
    register float fd1 __asm__(f1); \
    register float fd2 __asm__(f2); \
    register float fd3 __asm__(f3); \
    register float fd4 __asm__(f4); \
    register float fd5 __asm__(f5); \
    register float fd6 __asm__(f6); \
    register float fd7 __asm__(f7); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd0) : "m"(base[0])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd1) : "m"(base[1])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd2) : "m"(base[2])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd3) : "m"(base[3])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd4) : "m"(base[4])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd5) : "m"(base[5])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd6) : "m"(base[6])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd7) : "m"(base[7])); \
    ret = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7}; \
    return ret

#define MAKE_VX_WLDM_T_F32(f0, f1, f2, f3, f4, f5, f6, f7) \
    mf32x8_t ret; \
    auto base = (const float*)src + col; \
    register float fd0 __asm__(f0); \
    register float fd1 __asm__(f1); \
    register float fd2 __asm__(f2); \
    register float fd3 __asm__(f3); \
    register float fd4 __asm__(f4); \
    register float fd5 __asm__(f5); \
    register float fd6 __asm__(f6); \
    register float fd7 __asm__(f7); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd0) : "m"(base[0 * ldm])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd1) : "m"(base[1 * ldm])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd2) : "m"(base[2 * ldm])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd3) : "m"(base[3 * ldm])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd4) : "m"(base[4 * ldm])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd5) : "m"(base[5 * ldm])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd6) : "m"(base[6 * ldm])); \
    __asm__ volatile ("flw %0, %1" : "=f"(fd7) : "m"(base[7 * ldm])); \
    ret = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7}; \
    return ret

__attribute__((always_inline)) mf32x8_t vx_wldm_ad_f32(const void* src, int row, size_t ldm) {
    MAKE_VX_WLDM_D_F32("f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15");
}

__attribute__((always_inline)) mf32x8_t vx_wldm_at_f32(const void* src, int col, size_t ldm) {
    MAKE_VX_WLDM_T_F32("f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15");
}

__attribute__((always_inline)) mf32x8_t vx_wldm_bd_f32(const void* src, int row, size_t ldm) {
    MAKE_VX_WLDM_D_F32("f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
}

__attribute__((always_inline)) mf32x8_t vx_wldm_bt_f32(const void* src, int col, size_t ldm) {
    MAKE_VX_WLDM_T_F32("f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
}

__attribute__((always_inline)) void vx_wstm_f32(void* dst, const mf32x8_t& src, int row, int col, size_t ldm) {
    mf32x8_t ret;
    auto base = (float*)dst + row * ldm + col;
    auto base_2row = base + 2 * ldm;
    __asm__ volatile("fsw %0, %1" ::"f"(src[0]), "m"(base[0]));
    __asm__ volatile("fsw %0, %1" ::"f"(src[1]), "m"(base[1]));
    __asm__ volatile("fsw %0, %1" ::"f"(src[2]), "m"(base_2row[0]));
    __asm__ volatile("fsw %0, %1" ::"f"(src[3]), "m"(base_2row[1]));
    __asm__ volatile("fsw %0, %1" ::"f"(src[4]), "m"(base[4]));
    __asm__ volatile("fsw %0, %1" ::"f"(src[5]), "m"(base[5]));
    __asm__ volatile("fsw %0, %1" ::"f"(src[6]), "m"(base_2row[4]));
    __asm__ volatile("fsw %0, %1" ::"f"(src[7]), "m"(base_2row[5]));
}

#define MAKE_VX_HMMA_844_D_F32_STEP(fmt, step, rd_lo, rd_hi, rs1, rs2, rs3_lo, rs3_hi) \
    __asm__ volatile (".word %1" : "=r"(rd_lo) : "i"(RISCV_INSN_R(RISCV_CUSTOM0, 0, 2, fmt, step * 2 + 0, 1)), "r"(rs1), "r"(rs2), "r"(rs3_lo)); \
    __asm__ volatile (".word %1" : "=r"(rd_hi) : "i"(RISCV_INSN_R(RISCV_CUSTOM0, 0, 2, fmt, step * 2 + 1, 1)), "r"(rs1), "r"(rs2), "r"(rs3_hi))

#define MAKE_VX_HMMA_844_D_F32(fmt) \
    mf32x8_t ret; \
    register float fd0 __asm__("f16"); \
    register float fd1 __asm__("f17"); \
    register float fd2 __asm__("f18"); \
    register float fd3 __asm__("f19"); \
    register float fd4 __asm__("f20"); \
    register float fd5 __asm__("f21"); \
    register float fd6 __asm__("f22"); \
    register float fd7 __asm__("f23"); \
    register float fa0 __asm__("f8")  = a[0]; \
    register float fa1 __asm__("f9" ) = a[1]; \
    register float fa2 __asm__("f10") = a[2]; \
    register float fa3 __asm__("f11") = a[3]; \
    register float fa4 __asm__("f12") = a[4]; \
    register float fa5 __asm__("f13") = a[5]; \
    register float fa6 __asm__("f14") = a[6]; \
    register float fa7 __asm__("f15") = a[7]; \
    register float fb0 __asm__("f24") = b[0]; \
    register float fb1 __asm__("f25") = b[1]; \
    register float fb2 __asm__("f26") = b[2]; \
    register float fb3 __asm__("f27") = b[3]; \
    register float fb4 __asm__("f28") = b[4]; \
    register float fb5 __asm__("f29") = b[5]; \
    register float fb6 __asm__("f30") = b[6]; \
    register float fb7 __asm__("f31") = b[7]; \
    register float fc0 __asm__("f0")  = c[0]; \
    register float fc1 __asm__("f1")  = c[1]; \
    register float fc2 __asm__("f2")  = c[2]; \
    register float fc3 __asm__("f3")  = c[3]; \
    register float fc4 __asm__("f4")  = c[4]; \
    register float fc5 __asm__("f5")  = c[5]; \
    register float fc6 __asm__("f6")  = c[6]; \
    register float fc7 __asm__("f7")  = c[7]; \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 0,  fd0, fd1, fa0, fb0, fc0, fc1); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 1,  fd2, fd3, fa0, fb0, fc2, fc3); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 2,  fd4, fd5, fa0, fb0, fc4, fc5); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 3,  fd6, fd7, fa0, fb0, fc6, fc7); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 4,  fd0, fd1, fa1, fb1, fc0, fc1); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 5,  fd2, fd3, fa1, fb1, fc2, fc3); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 6,  fd4, fd5, fa1, fb1, fc4, fc5); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 7,  fd6, fd7, fa1, fb1, fc6, fc7); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 8,  fd0, fd1, fa2, fb2, fc0, fc1); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 9,  fd2, fd3, fa2, fb2, fc2, fc3); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 10, fd4, fd5, fa2, fb2, fc4, fc5); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 11, fd6, fd7, fa2, fb2, fc6, fc7); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 12, fd0, fd1, fa3, fb3, fc0, fc1); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 13, fd2, fd3, fa3, fb3, fc2, fc3); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 14, fd4, fd5, fa3, fb3, fc4, fc5); \
    MAKE_VX_HMMA_844_D_F32_STEP(fmt, 15, fd6, fd7, fa3, fb3, fc6, fc7); \
    ret = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7}; \
    return ret

#define MAKE_VX_HMMA_844_C_F32_STEP(fmt, step, rd_lo, rd_hi, rs1, rs2) \
    __asm__ volatile (".word %1" : "=r"(rd_lo) : "i"(RISCV_INSN_R(RISCV_CUSTOM0, 0, 2, fmt, step * 2 + 0, 0)), "r"(rs1), "r"(rs2), "r"(rd_lo)); \
    __asm__ volatile (".word %1" : "=r"(rd_hi) : "i"(RISCV_INSN_R(RISCV_CUSTOM0, 0, 2, fmt, step * 2 + 1, 0)), "r"(rs1), "r"(rs2), "r"(rd_hi))

#define MAKE_VX_HMMA_844_C_F32(fmt) \
    mf32x8_t ret; \
    register float fa0 __asm__("f8")  = a[0]; \
    register float fa1 __asm__("f9" ) = a[1]; \
    register float fa2 __asm__("f10") = a[2]; \
    register float fa3 __asm__("f11") = a[3]; \
    register float fa4 __asm__("f12") = a[4]; \
    register float fa5 __asm__("f13") = a[5]; \
    register float fa6 __asm__("f14") = a[6]; \
    register float fa7 __asm__("f15") = a[7]; \
    register float fb0 __asm__("f24") = b[0]; \
    register float fb1 __asm__("f25") = b[1]; \
    register float fb2 __asm__("f26") = b[2]; \
    register float fb3 __asm__("f27") = b[3]; \
    register float fb4 __asm__("f28") = b[4]; \
    register float fb5 __asm__("f29") = b[5]; \
    register float fb6 __asm__("f30") = b[6]; \
    register float fb7 __asm__("f31") = b[7]; \
    register float fc0 __asm__("f0")  = c[0]; \
    register float fc1 __asm__("f1")  = c[1]; \
    register float fc2 __asm__("f2")  = c[2]; \
    register float fc3 __asm__("f3")  = c[3]; \
    register float fc4 __asm__("f4")  = c[4]; \
    register float fc5 __asm__("f5")  = c[5]; \
    register float fc6 __asm__("f6")  = c[6]; \
    register float fc7 __asm__("f7")  = c[7]; \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 0,  fc0, fc1, fa0, fb0); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 1,  fc2, fc3, fa0, fb0); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 2,  fc4, fc5, fa0, fb0); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 3,  fc6, fc7, fa0, fb0); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 4,  fc0, fc1, fa1, fb1); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 5,  fc2, fc3, fa1, fb1); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 6,  fc4, fc5, fa1, fb1); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 7,  fc6, fc7, fa1, fb1); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 8,  fc0, fc1, fa2, fb2); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 9,  fc2, fc3, fa2, fb2); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 10, fc4, fc5, fa2, fb2); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 11, fc6, fc7, fa2, fb2); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 12, fc0, fc1, fa3, fb3); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 13, fc2, fc3, fa3, fb3); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 14, fc4, fc5, fa3, fb3); \
    MAKE_VX_HMMA_844_C_F32_STEP(fmt, 15, fc6, fc7, fa3, fb3); \
    ret = {fc0, fc1, fc2, fc3, fc4, fc5, fc6, fc7}; \
    return ret

__attribute__((always_inline)) mf32x8_t vx_hmma_844_c_f16_f32(const mf32x8_t& a, const mf32x8_t& b, const mf32x8_t& c) {
    MAKE_VX_HMMA_844_C_F32(0);
}

__attribute__((always_inline)) mf32x8_t vx_hmma_844_d_f16_f32(const mf32x8_t& a, const mf32x8_t& b, const mf32x8_t& c) {
    MAKE_VX_HMMA_844_D_F32(0);
}

#ifdef __cplusplus
}
#endif

#endif // __VX_INTRINSICS_H__
