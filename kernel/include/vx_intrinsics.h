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

#ifndef __VX_INTRINSICS_H__
#define __VX_INTRINSICS_H__

#include <stddef.h>
#include <VX_types.h>

#if defined(__clang__)
#define __UNIFORM__   __attribute__((annotate("vortex.uniform")))
#else
#define __UNIFORM__
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __ASSEMBLY__
#define __ASM_STR(x)	x
#else
#define __ASM_STR(x)	#x
#endif

#define RISCV_CUSTOM0   0x0B
#define RISCV_CUSTOM1   0x2B
#define RISCV_CUSTOM2   0x5B
#define RISCV_CUSTOM3   0x7B

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
inline void vx_pred(int condition, int thread_mask) {
    asm volatile (".insn r %0, 5, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask));
}

// Set thread not predicate
inline void vx_pred_n(int condition, int thread_mask) {
    asm volatile (".insn r %0, 5, 0, x1, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask));
}

typedef void (*vx_wspawn_pfn)();

// Spawn warps
inline void vx_wspawn(size_t num_warps, vx_wspawn_pfn func_ptr) {
    asm volatile (".insn r %0, 1, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(num_warps), "r"(func_ptr));
}

// Split on a predicate
inline int vx_split(int predicate) {
    size_t ret;
    asm volatile (".insn r %1, 2, 0, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Split on a not predicate
inline int vx_split_n(int predicate) {
    size_t ret;
    asm volatile (".insn r %1, 2, 0, %0, %2, x1" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Join
inline void vx_join(int stack_ptr) {
    asm volatile (".insn r %0, 3, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(stack_ptr));
}

// Warp Barrier
inline void vx_barrier(int barried_id, int num_warps) {
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

// Return current thread mask
inline int vx_thread_mask() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_THREAD_MASK));
    return ret;
}

// Return active warps mask
inline int vx_warp_mask() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_WARP_MASK));
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

// Return the hart identifier (thread id accross the processor)
inline int vx_hart_id() {
    int ret;
    asm volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_MHARTID));
    return ret;
}

inline void vx_fence() {
    asm volatile ("fence iorw, iorw");
}

#ifdef __cplusplus
}
#endif

#endif // __VX_INTRINSICS_H__
