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

#define csr_read(csr) ({                        \
	size_t __r;	               		            \
	__asm__ volatile ("csrr %0, %1" : "=r" (__r) : "i" (csr)); \
	__r;							            \
})

// Non-volatile CSR read: for CSRs that are constant over the reader's lifetime
// (e.g. CTA identity/dimension), so the compiler may CSE/hoist them out of loops.
// Use csr_read (volatile) for CSRs that change across reads, e.g. mcycle.
#define csr_read_nv(csr) ({                     \
	size_t __r;	               		            \
	__asm__ ("csrr %0, %1" : "=r" (__r) : "i" (csr)); \
	__r;							            \
})

#define csr_write(csr, val)	({                  \
	size_t __v = (size_t)(val);                 \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ volatile ("csrwi %0, %1" :: "i" (csr), "i" (__v));  \
    else                                        \
        __asm__ volatile ("csrw %0, %1"	:: "i" (csr), "r" (__v));  \
})

#define csr_swap(csr, val) ({                   \
    size_t __r;                                 \
	size_t __v = (size_t)(val);	                \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ volatile ("csrrwi %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ volatile ("csrrw %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v)); \
	__r;						                \
})

#define csr_read_set(csr, val) ({               \
	size_t __r;                                 \
	size_t __v = (size_t)(val);	                \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ volatile ("csrrsi %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ volatile ("csrrs %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v)); \
	__r;							            \
})

#define csr_set(csr, val) ({                    \
	size_t __v = (size_t)(val);	                \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ volatile ("csrsi %0, %1" :: "i" (csr), "i" (__v));  \
    else                                        \
        __asm__ volatile ("csrs %0, %1"	:: "i" (csr), "r" (__v));  \
})

#define csr_read_clear(csr, val) ({             \
	size_t __r;                                 \
	size_t __v = (size_t)(val);	                \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ volatile ("csrrci %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ volatile ("csrrc %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v)); \
	__r;							            \
})

#define csr_clear(csr, val)	({                  \
	size_t __v = (size_t)(val);                 \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ volatile ("csrci %0, %1" :: "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ volatile ("csrc %0, %1"	:: "i" (csr), "r" (__v)); \
})

// Trap return: restore PC from `mepc` and pop the M-mode trap state.
// For use at the bottom of a trap/callback dispatcher.
#define vx_mret() \
  __asm__ volatile ("mret")

// Set thread mask
inline void vx_tmc(int thread_mask) {
    __asm__ volatile (".insn r %0, 0, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(thread_mask) : "memory");
}

// Disable all threads in the current warp.
inline void vx_tmc_zero() {
    __asm__ volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM0) : "memory");
}

// Switch execution to thread 0 only.
inline void vx_tmc_one() {
    __asm__ volatile (
        "li a0, 1\n\t"
        ".insn r %0, 0, 0, x0, a0, x0" :: "i"(RISCV_CUSTOM0) : "a0", "memory"
    );
}

// Set thread predicate
inline void vx_pred(int condition, int thread_mask) {
    __asm__ volatile (".insn r %0, 5, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask) : "memory");
}

// Set thread not predicate
inline void vx_pred_n(int condition, int thread_mask) {
    __asm__ volatile (".insn r %0, 5, 0, x1, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(condition), "r"(thread_mask) : "memory");
}

// Spawn warps
typedef void (*vx_wspawn_pfn)();
inline void vx_wspawn(int num_warps, vx_wspawn_pfn func_ptr) {
    __asm__ volatile (".insn r %0, 1, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(num_warps), "r"(func_ptr) : "memory");
}

// Split on a predicate
inline int vx_split(int predicate) {
    int ret;
    __asm__ volatile (".insn r %1, 2, 0, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate) : "memory");
    return ret;
}

// Split on a not predicate
inline int vx_split_n(int predicate) {
    int ret;
    __asm__ volatile (".insn r %1, 2, 0, %0, %2, x1" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate) : "memory");
    return ret;
}

// Join
inline void vx_join(int stack_ptr) {
    __asm__ volatile (".insn r %0, 3, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(stack_ptr) : "memory");
}

// Warp Barrier
inline void vx_barrier(int barried_id, int num_warps) {
    __asm__ volatile (".insn r %0, 4, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(barried_id), "r"(num_warps) : "memory");
}

// Return current thread identifier (constant for the thread's lifetime: const +
// csr_read_nv so the compiler may CSE/hoist it)
inline __attribute__((const)) int vx_thread_id() {
    return (int)csr_read_nv(VX_CSR_THREAD_ID);
}

// Return current warp identifier
inline __attribute__((const)) int vx_warp_id() {
    return (int)csr_read_nv(VX_CSR_WARP_ID);
}

// Return current core identifier
inline __attribute__((const)) int vx_core_id() {
    return (int)csr_read_nv(VX_CSR_CORE_ID);
}

// Return active threads mask (changes with divergence: NOT const, volatile csr_read)
inline size_t vx_active_threads() {
    return csr_read(VX_CSR_ACTIVE_THREADS);
}

// Return active warps mask (changes with divergence: NOT const, volatile csr_read)
inline size_t vx_active_warps() {
    return csr_read(VX_CSR_ACTIVE_WARPS);
}

// Return the number of threads per warp
inline __attribute__((const)) int vx_num_threads() {
    return (int)csr_read_nv(VX_CSR_NUM_THREADS);
}

// Return the number of warps per core
inline __attribute__((const)) int vx_num_warps() {
    return (int)csr_read_nv(VX_CSR_NUM_WARPS);
}

// Return the number of cores across all clusters
inline __attribute__((const)) int vx_num_cores() {
    return (int)csr_read_nv(VX_CSR_NUM_CORES);
}

// Return the number of barriers
inline __attribute__((const)) int vx_num_barriers() {
    return (int)csr_read_nv(VX_CSR_NUM_BARRIERS);
}

// Return the hart identifier (thread id accross the processor)
inline __attribute__((const)) int vx_hart_id() {
    return (int)csr_read_nv(VX_CSR_MHARTID);
}

//
// profiling
//

// Return current cycle counter
inline uint64_t vx_rdcycle() {
#if __riscv_xlen == 64
    return csr_read(VX_CSR_MCYCLE);
#elif __riscv_xlen == 32
    uint32_t hi0, lo, hi1;
    do {
        hi0 = csr_read(VX_CSR_MCYCLE_H);
        lo  = csr_read(VX_CSR_MCYCLE);
        hi1 = csr_read(VX_CSR_MCYCLE_H);
    } while (hi0 != hi1);
    return (((uint64_t)hi0) << 32) | lo;
#else
#error "Unsupported RISC-V XLEN"
#endif
}

// Warp Sync
inline void vx_wsync() {
    __asm__ volatile (".insn r %0, 7, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM0) : "memory");
}


/* Safely flushes the warp pipeline and reads the 64-bit cycle counter.
 * Automatically handles 32-bit overflow mitigation or native 64-bit reads.
 */
static inline uint64_t vx_rdcycle_sync() {
    uint64_t cycles;
#if __riscv_xlen == 32
    uint32_t cycle_lo, cycle_hi, cycle_hi_check;
    __asm__ volatile (
        ".insn r %3, 7, 0, x0, x0, x0\n\t"
        "1:\n\t"
        "csrr %0, %4\n\t"
        "csrr %1, %5\n\t"
        "csrr %2, %4\n\t"
        "bne %0, %2, 1b\n\t"
        : "=&r" (cycle_hi), "=&r" (cycle_lo), "=&r" (cycle_hi_check)
        : "i" (RISCV_CUSTOM0), "i" (VX_CSR_MCYCLE_H), "i" (VX_CSR_MCYCLE)
        : "memory"
    );
    cycles = ((uint64_t)cycle_hi << 32) | cycle_lo;
#elif __riscv_xlen == 64
    __asm__ volatile (
        ".insn r %1, 7, 0, x0, x0, x0\n\t"
        "csrr %0, %2\n\t"
        : "=r" (cycles)
        : "i" (RISCV_CUSTOM0), "i" (VX_CSR_MCYCLE)
        : "memory"
    );
#else
#error "Unsupported RISC-V XLEN"
#endif

    return cycles;
}

typedef struct {
    uint32_t hi;
    uint32_t lo;
} __rdcycle_time;

static inline __attribute__((always_inline)) __rdcycle_time vx_rdcycle_sync_begin(void) {
    __rdcycle_time t;
#if __riscv_xlen == 32
    __asm__ volatile (
        ".insn r %2, 7, 0, x0, x0, x0\n\t"
        "csrr %0, %3\n\t"
        "csrr %1, %4\n\t"
        : "=r" (t.hi), "=r" (t.lo)
        : "i" (RISCV_CUSTOM0), "i" (VX_CSR_MCYCLE_H), "i" (VX_CSR_MCYCLE)
        : "memory"
    );
#elif __riscv_xlen == 64
    uint64_t cycles;
    __asm__ volatile (
        ".insn r %1, 7, 0, x0, x0, x0\n\t"
        "csrr %0, %2\n\t"
        : "=r" (cycles)
        : "i" (RISCV_CUSTOM0), "i" (VX_CSR_MCYCLE)
        : "memory"
    );
    t.hi = (uint32_t)(cycles >> 32);
    t.lo = (uint32_t)cycles;
#else
#error "Unsupported RISC-V XLEN"
#endif
    return t;
}

static inline __attribute__((always_inline)) __rdcycle_time vx_rdcycle_sync_end(void) {
    __rdcycle_time t;
#if __riscv_xlen == 32
    __asm__ volatile (
        ".insn r %2, 7, 0, x0, x0, x0\n\t"
        "csrr %0, %3\n\t"
        "csrr %1, %4\n\t"
        : "=r" (t.lo), "=r" (t.hi)
        : "i" (RISCV_CUSTOM0), "i" (VX_CSR_MCYCLE), "i" (VX_CSR_MCYCLE_H)
        : "memory"
    );
#elif __riscv_xlen == 64
    uint64_t cycles;
    __asm__ volatile (
        ".insn r %1, 7, 0, x0, x0, x0\n\t"
        "csrr %0, %2\n\t"
        : "=r" (cycles)
        : "i" (RISCV_CUSTOM0), "i" (VX_CSR_MCYCLE)
        : "memory"
    );
    t.hi = (uint32_t)(cycles >> 32);
    t.lo = (uint32_t)cycles;
#else
#error "Unsupported RISC-V XLEN"
#endif
    return t;
}

static inline __attribute__((always_inline)) uint64_t vx_rdcycle_sync_diff(__rdcycle_time start, __rdcycle_time end) {
#if __riscv_xlen == 32
    uint32_t diff_hi = end.hi;
    uint32_t diff_lo = end.lo;
    uint32_t tmp = start.hi;
    uint32_t start_lo = start.lo;

    __asm__ volatile (
        "sub  %0, %0, %2\n\t"
        "sltu %2, %1, %3\n\t"
        "sub  %1, %1, %3\n\t"
        "sub  %0, %0, %2\n\t"
        : "+r" (diff_hi), "+r" (diff_lo), "+r" (tmp)
        : "r" (start_lo)
        : "memory"
    );

    return ((uint64_t)diff_hi << 32) | diff_lo;
#elif __riscv_xlen == 64
    uint64_t s = ((uint64_t)start.hi << 32) | start.lo;
    uint64_t e = ((uint64_t)end.hi << 32) | end.lo;
    return e - s;
#else
#error "Unsupported RISC-V XLEN"
#endif
}

// Memory fence
inline void vx_fence() {
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
}


//
// cooperative threads extensions
//

// Returns 1 if every active lane’s predicate is true, 0 otherwise.
inline __attribute__((const)) size_t vx_vote_all(int predicate) {
    size_t ret;
    __asm__ volatile (".insn r %1, 0, 1, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Returns 1 if any active lane’s predicate is true, 0 if none are true.
inline __attribute__((const)) size_t vx_vote_any(int predicate) {
    size_t ret;
    __asm__ volatile (".insn r %1, 1, 1, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

//  Returns 1 if the predicate is uniform across all active lanes.
inline __attribute__((const)) size_t vx_vote_uni(int predicate) {
    size_t ret;
    __asm__ volatile (".insn r %1, 2, 1, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Returns a bitmask of the warp, with bit i set if lane i’s predicate is true.
inline __attribute__((const)) size_t vx_vote_ballot(int predicate) {
    size_t ret;
    __asm__ volatile (".insn r %1, 3, 1, %0, %2, x0" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(predicate));
    return ret;
}

// Shift values up by b lanes within each sub-group; out-of-range lanes keep their own value.
inline __attribute__((const)) size_t vx_shfl_up(size_t value, int bval, int cval, int mask) {
    size_t ret;
    int bc = (mask << 12) | (cval << 6) | bval;
    __asm__ volatile (".insn r %1, 4, 1, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(value), "r"(bc));
    return ret;
}

// Shift values down by b lanes within each sub-group; out-of-range lanes keep their own value.
inline __attribute__((const)) size_t vx_shfl_down(size_t value, int bval, int cval, int mask) {
    size_t ret;
    int bc = (mask << 12) | (cval << 6) | bval;
    __asm__ volatile (".insn r %1, 5, 1, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(value), "r"(bc));
    return ret;
}

// Segment operands that scope a shuffle to a group of four adjacent lanes:
// lane l's group is [l & ~3, (l & ~3) | 3]. The mask is spelled at the full
// 6-bit operand width; the hardware truncates it to the lane-index width, and
// ~3 truncates to ~3 at every width from 2 to 6 bits.
//
// A shuffle whose source lane is inactive returns the reader's own value, so a
// group's lanes must all be active for a cross-lane read to see its neighbours.
#define VX_QUAD_CVAL 3
#define VX_QUAD_MASK 0x3c

// “Butterfly” exchange using XOR with b as a bit‐mask: each lane swaps with lane ⊕ b.
inline __attribute__((const)) size_t vx_shfl_bfly(size_t value, int bval, int cval, int mask) {
    size_t ret;
    int bc = (mask << 12) | (cval << 6) | bval;
    __asm__ volatile (".insn r %1, 6, 1, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(value), "r"(bc));
    return ret;
}

// Gather from an explicit index: every lane reads the value from base + idx, where idx = b[i].
inline __attribute__((const)) size_t vx_shfl_idx(size_t value, int bval, int cval, int mask) {
    size_t ret;
    int bc = (mask << 12) | (cval << 6) | bval;
    __asm__ volatile (".insn r %1, 7, 1, %0, %2, %3" : "=r"(ret) : "i"(RISCV_CUSTOM0), "r"(value), "r"(bc));
    return ret;
}

//
// Quad ops
//
// A pixel quad occupies four adjacent lanes, so lane l's horizontal neighbour is
// l^1 and its vertical neighbour is l^2. A derivative is the difference against
// that neighbour, which is why derivatives are undefined when a quad diverges:
// an inactive source lane returns the reader's own value, so the difference
// collapses to zero rather than faulting.

// Exchange with a quad neighbour: dir 1 = horizontal, 2 = vertical.
inline __attribute__((const)) size_t vx_quad_swap(size_t value, int dir) {
    return vx_shfl_bfly(value, dir, VX_QUAD_CVAL, VX_QUAD_MASK);
}

// Read quad lane `idx` (0..3); every lane of the quad gets the same value.
inline __attribute__((const)) size_t vx_quad_bcast(size_t value, int idx) {
    return vx_shfl_idx(value, idx, VX_QUAD_CVAL, VX_QUAD_MASK);
}

// The exchange is symmetric, so a bare (neighbour - self) would hand the two lanes
// of a pair opposite signs. A derivative points the same way for the whole quad --
// towards increasing x for ddx, increasing y for ddy -- so the lane holding the FAR
// pixel of the pair subtracts the other way. (A caller that only wants the magnitude
// is free to skip this, but the sign is what makes these derivatives.)
inline __attribute__((const)) int32_t vx_quad_ddx_i32(int32_t value) {
    int32_t neighbor = (int32_t)vx_quad_swap((size_t)value, 1);
    return (vx_thread_id() & 1) ? (value - neighbor) : (neighbor - value);
}

inline __attribute__((const)) int32_t vx_quad_ddy_i32(int32_t value) {
    int32_t neighbor = (int32_t)vx_quad_swap((size_t)value, 2);
    return (vx_thread_id() & 2) ? (value - neighbor) : (neighbor - value);
}

inline __attribute__((const)) float vx_quad_ddx_f32(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    uint32_t nbits = (uint32_t)vx_quad_swap((size_t)bits, 1);
    float neighbor;
    __builtin_memcpy(&neighbor, &nbits, sizeof(neighbor));
    return (vx_thread_id() & 1) ? (value - neighbor) : (neighbor - value);
}

inline __attribute__((const)) float vx_quad_ddy_f32(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    uint32_t nbits = (uint32_t)vx_quad_swap((size_t)bits, 2);
    float neighbor;
    __builtin_memcpy(&neighbor, &nbits, sizeof(neighbor));
    return (vx_thread_id() & 2) ? (value - neighbor) : (neighbor - value);
}

//
// Warp-Level Lane Gather Extension
//

// Each lane gathers a value from the source lane's register file.
// S = source lane (compile-time constant, 0-3).
// The source lane retains its own rd value; lane (S+1) gets v1[S], (S+2) gets v2[S], (S+3) gets v3[S].
//
// WGATHER writes every non-source lane of rd, active or not, so a unit reading
// the packed operand across the warp (the RTU trace config, a DXA descriptor)
// sees all of it whatever the thread mask. In an allocatable register that
// would clobber an inactive lane's live value, so rd is pinned to a dedicated
// warp-gather register -- x31, or x30 for a second packed operand live at the
// same time -- which the compiler reserves in any function naming one. A
// consumer of the packed lanes must read that register directly (bind it with
// a register variable); a copy elsewhere is written on active lanes only.
#define __VX_WGATHER_IN(reg, src_lane, self_val, v1, v2, v3) ({ \
    register size_t __ret __asm__(reg) = (self_val);    \
    __asm__ volatile (                                  \
        ".insn r4 %1, 0, %2, %0, %3, %4, %5"            \
        : "+r"(__ret)                                   \
        : "i"(RISCV_CUSTOM1), "i"(src_lane),            \
          "r"(v1), "r"(v2), "r"(v3)                     \
    );                                                  \
    __ret;                                              \
})

#define __VX_WGATHER(src_lane, self_val, v1, v2, v3) \
    __VX_WGATHER_IN("x31", src_lane, self_val, v1, v2, v3)

// Warp-level gather with source lane 0.
inline __attribute__((const)) size_t
vx_wgather(size_t self_val, size_t v1, size_t v2, size_t v3) {
    return __VX_WGATHER(0, self_val, v1, v2, v3);
}

// Warp-level gather with an explicit (compile-time constant) source lane.
#define vx_wgather_from(src_lane, self_val, v1, v2, v3) \
    __VX_WGATHER(src_lane, self_val, v1, v2, v3)

// Transpose a 4×4 matrix distributed one row per lane, using 4 wgather instructions.
//
// Each lane i must hold one complete row of the matrix in four registers:
//   a0[i] = M[i][0],  a1[i] = M[i][1],  a2[i] = M[i][2],  a3[i] = M[i][3]
//
// After the macro, each lane i holds column i:
//   out0[i] = M[0][i],  out1[i] = M[1][i],  out2[i] = M[2][i],  out3[i] = M[3][i]
//
// Operates within each independent group of 4 lanes (scalable to any NUM_LANES).
// out0..out3 must be distinct lvalues from a0..a3 (not in-place safe).
#define vx_transpose4(a0, a1, a2, a3, out0, out1, out2, out3) do { \
    (out0) = vx_wgather_from(0, (a0), (a1), (a2), (a3));           \
    (out1) = vx_wgather_from(1, (a1), (a2), (a3), (a0));           \
    (out2) = vx_wgather_from(2, (a2), (a3), (a0), (a1));           \
    (out3) = vx_wgather_from(3, (a3), (a0), (a1), (a2));           \
} while (0)

//
// Asynchronous Barrier extensions
//

// Async Barrier Arrive: non-blocking, returns a phase (generation number)
// barrier_id: identifier of the barrier
// num_warps: number of warps participating in the barrier
// returns: number representing the barrier phase for this arrive
inline int vx_barrier_arrive(int barrier_id, int num_warps) {
    int phase;
    __asm__ volatile (
        ".insn r %1, 6, 0, %0, %2, %3" : "=r"(phase) : "i"(RISCV_CUSTOM0), "r"(barrier_id), "r"(num_warps) : "memory"
    );
    return phase;
}

// Async Barrier Wait: blocks until a barrier phase is complete
// barrier_id: identifier of the barrier
// phase: the phase returned by vx_barrier_arrive
inline void vx_barrier_wait(int barrier_id, int phase) {
    __asm__ volatile (
        ".insn r %0, 6, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), "r"(barrier_id), "r"(phase) : "memory"
    );
}

// Async Transaction Expect: pre-register N pending transaction events on
// the given barrier. Reuses the barrier_arrive opcode but encodes rs2[31]=1
// to flag expect-tx semantics. The lower bits of rs2 carry the count N.
// Each pending event must be matched by a future release (e.g. DXA SMEM
// completion). The barrier only flips phase when both arrive_count==0 AND
// events==0.
//
// NOTE: rd must be non-x0 — decode uses (rd != 0) to distinguish arrive
// vs wait on this opcode. The returned phase is meaningless for expect_tx
// (we don't advance arrival count) but we keep a real register destination
// to satisfy the decoder.
inline void vx_barrier_expect_tx(int barrier_id, int count) {
    int discard;
    int rs2 = count | 0x80000000;
    __asm__ volatile (
        ".insn r %1, 6, 0, %0, %2, %3" : "=r"(discard) : "i"(RISCV_CUSTOM0), "r"(barrier_id), "r"(rs2) : "memory"
    );
    (void)discard;
}

//
// Load Bytes Packing Extension
//

// Packed-load 4 bytes into one FP register bit-container.
// base: pointer to the first byte to load
// stride: byte offset between consecutive bytes to load
// fd = bitcast<float>((MEM8[base + 0 * stride] << 0)
//                   | (MEM8[base + 1 * stride] << 8)
//                   | (MEM8[base + 2 * stride] << 16)
//                   | (MEM8[base + 3 * stride] << 24));
__attribute__((always_inline))
inline float vx_packlb_f(const void* base, size_t stride) {
    float out;
    __asm__ volatile (
        ".insn r %1, 1, 4, %0, %2, %3" : "=f"(out) : "i"(RISCV_CUSTOM0), "r"(base), "r"(stride) : "memory"
    );
    return out;
}

// Packed-load 2 halfwords into one FP register bit-container.
// base: pointer to the first halfword to load
// stride: byte offset between consecutive halfwords to load
// fd = bitcast<float>((MEM16[base + 0 * stride] << 0)
//                   | (MEM16[base + 1 * stride] << 16));
__attribute__((always_inline))
inline float vx_packlh_f(const void* base, size_t stride) {
    float out;
    __asm__ volatile (
        ".insn r %1, 2, 4, %0, %2, %3" : "=f"(out) : "i"(RISCV_CUSTOM0), "r"(base), "r"(stride) : "memory"
    );
    return out;
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#if defined(VX_CFG_EXT_LDST_WIDE_ENABLE) || defined(VX_CFG_EXT_LDST_MATRIX_ENABLE)
// FP register lists for the register-pinned operands of the wide and matrix
// loads and stores; a list expands before the emitter splits it into names.
#define __VX_FREGS2(a, b)                   "f" #a, "f" #b
#define __VX_FREGS4(a, b, c, d)             "f" #a, "f" #b, "f" #c, "f" #d
#define __VX_FREGS8(a, b, c, d, e, f, g, h) "f" #a, "f" #b, "f" #c, "f" #d, \
                                            "f" #e, "f" #f, "f" #g, "f" #h
#endif

#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE

//
// Wide loads and stores
//

// vx_flw.d/.q fd, OFF(addr) and vx_fsw.d/.q fs2, OFF(addr): 2 or 4 consecutive
// words, each 4-byte aligned, to or from the FP register group fd = FD (a
// multiple of 2 or 4); OFF is a multiple of 8 or 16. FD only binds the asm
// operands: values live at once in the same group stay correct but cost
// register copies, so they are best given different FD.
typedef float vx_float2 __attribute__((vector_size(8), aligned(4)));
typedef float vx_float4 __attribute__((vector_size(16), aligned(4)));

#define __VX_FLW(funct3, ...) __VX_FLW_(funct3, __VA_ARGS__)
#define __VX_FLW_(funct3, r0, r1, ...) {                                              \
    register float x0 __asm__(r0); register float x1 __asm__(r1);                     \
    __asm__ volatile (".insn i %[op], " #funct3 ", %[x0], %[a], %[off]"              \
      : [x0]"=f"(x0), "=f"(x1)                                                       \
      : [op]"i"(RISCV_CUSTOM3), [a]"r"(addr), [off]"i"(OFF) : "memory");            \
    return vx_float2{x0, x1}; }
#define __VX_FLW4(funct3, ...) __VX_FLW4_(funct3, __VA_ARGS__)
#define __VX_FLW4_(funct3, r0, r1, r2, r3, ...) {                                     \
    register float x0 __asm__(r0); register float x1 __asm__(r1);                     \
    register float x2 __asm__(r2); register float x3 __asm__(r3);                     \
    __asm__ volatile (".insn i %[op], " #funct3 ", %[x0], %[a], %[off]"              \
      : [x0]"=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)                                   \
      : [op]"i"(RISCV_CUSTOM3), [a]"r"(addr), [off]"i"(OFF) : "memory");            \
    return vx_float4{x0, x1, x2, x3}; }
#define __VX_FSW(funct3, ...) __VX_FSW_(funct3, __VA_ARGS__)
#define __VX_FSW_(funct3, r0, r1, ...) {                                              \
    register float x0 __asm__(r0) = value[0]; register float x1 __asm__(r1) = value[1]; \
    __asm__ volatile (".insn s %[op], " #funct3 ", %[x0], %[off](%[a])"             \
      :: [op]"i"(RISCV_CUSTOM3), [a]"r"(addr), [off]"i"(OFF), [x0]"f"(x0), "f"(x1)  \
      : "memory"); }
#define __VX_FSW4(funct3, ...) __VX_FSW4_(funct3, __VA_ARGS__)
#define __VX_FSW4_(funct3, r0, r1, r2, r3, ...) {                                     \
    register float x0 __asm__(r0) = value[0]; register float x1 __asm__(r1) = value[1]; \
    register float x2 __asm__(r2) = value[2]; register float x3 __asm__(r3) = value[3]; \
    __asm__ volatile (".insn s %[op], " #funct3 ", %[x0], %[off](%[a])"             \
      :: [op]"i"(RISCV_CUSTOM3), [a]"r"(addr), [off]"i"(OFF), [x0]"f"(x0), "f"(x1),  \
         "f"(x2), "f"(x3) : "memory"); }

#define __VX_WIDE_Q(EMIT, ...) \
  if constexpr (FD == 0) { EMIT(__VX_FREGS4(0,1,2,3), __VA_ARGS__) } else \
  if constexpr (FD == 4) { EMIT(__VX_FREGS4(4,5,6,7), __VA_ARGS__) } else \
  if constexpr (FD == 8) { EMIT(__VX_FREGS4(8,9,10,11), __VA_ARGS__) } else \
  if constexpr (FD == 12) { EMIT(__VX_FREGS4(12,13,14,15), __VA_ARGS__) } else \
  if constexpr (FD == 16) { EMIT(__VX_FREGS4(16,17,18,19), __VA_ARGS__) } else \
  if constexpr (FD == 20) { EMIT(__VX_FREGS4(20,21,22,23), __VA_ARGS__) } else \
  if constexpr (FD == 24) { EMIT(__VX_FREGS4(24,25,26,27), __VA_ARGS__) } else \
  if constexpr (FD == 28) { EMIT(__VX_FREGS4(28,29,30,31), __VA_ARGS__) } else \
  { static_assert(FD % 4 == 0 && FD < 32, "a quad's fd is a multiple of 4"); }

#define __VX_WIDE_D(EMIT, ...) \
  if constexpr (FD == 0) { EMIT(__VX_FREGS2(0,1), __VA_ARGS__) } else \
  if constexpr (FD == 2) { EMIT(__VX_FREGS2(2,3), __VA_ARGS__) } else \
  if constexpr (FD == 4) { EMIT(__VX_FREGS2(4,5), __VA_ARGS__) } else \
  if constexpr (FD == 6) { EMIT(__VX_FREGS2(6,7), __VA_ARGS__) } else \
  if constexpr (FD == 8) { EMIT(__VX_FREGS2(8,9), __VA_ARGS__) } else \
  if constexpr (FD == 10) { EMIT(__VX_FREGS2(10,11), __VA_ARGS__) } else \
  if constexpr (FD == 12) { EMIT(__VX_FREGS2(12,13), __VA_ARGS__) } else \
  if constexpr (FD == 14) { EMIT(__VX_FREGS2(14,15), __VA_ARGS__) } else \
  if constexpr (FD == 16) { EMIT(__VX_FREGS2(16,17), __VA_ARGS__) } else \
  if constexpr (FD == 18) { EMIT(__VX_FREGS2(18,19), __VA_ARGS__) } else \
  if constexpr (FD == 20) { EMIT(__VX_FREGS2(20,21), __VA_ARGS__) } else \
  if constexpr (FD == 22) { EMIT(__VX_FREGS2(22,23), __VA_ARGS__) } else \
  if constexpr (FD == 24) { EMIT(__VX_FREGS2(24,25), __VA_ARGS__) } else \
  if constexpr (FD == 26) { EMIT(__VX_FREGS2(26,27), __VA_ARGS__) } else \
  if constexpr (FD == 28) { EMIT(__VX_FREGS2(28,29), __VA_ARGS__) } else \
  if constexpr (FD == 30) { EMIT(__VX_FREGS2(30,31), __VA_ARGS__) } else \
  { static_assert(FD % 2 == 0 && FD < 32, "a pair's fd is a multiple of 2"); }

#define __VX_EMIT_FLW_D(...)  __VX_FLW(3, __VA_ARGS__)
#define __VX_EMIT_FLW_Q(...)  __VX_FLW4(4, __VA_ARGS__)
#define __VX_EMIT_FSW_D(...)  __VX_FSW(5, __VA_ARGS__)
#define __VX_EMIT_FSW_Q(...)  __VX_FSW4(6, __VA_ARGS__)

template <uint32_t FD = 30, int32_t OFF = 0>
__attribute__((always_inline)) inline vx_float2 vx_flw_d(const void* addr) {
  static_assert(OFF % 8 == 0 && OFF >= -2048 && OFF <= 2040, "a pair's offset is a multiple of 8");
  __VX_WIDE_D(__VX_EMIT_FLW_D, _)
}

template <uint32_t FD = 28, int32_t OFF = 0>
__attribute__((always_inline)) inline vx_float4 vx_flw_q(const void* addr) {
  static_assert(OFF % 16 == 0 && OFF >= -2048 && OFF <= 2032, "a quad's offset is a multiple of 16");
  __VX_WIDE_Q(__VX_EMIT_FLW_Q, _)
}

template <uint32_t FD = 30, int32_t OFF = 0>
__attribute__((always_inline)) inline void vx_fsw_d(void* addr, vx_float2 value) {
  static_assert(OFF % 8 == 0 && OFF >= -2048 && OFF <= 2040, "a pair's offset is a multiple of 8");
  __VX_WIDE_D(__VX_EMIT_FSW_D, _)
}

template <uint32_t FD = 28, int32_t OFF = 0>
__attribute__((always_inline)) inline void vx_fsw_q(void* addr, vx_float4 value) {
  static_assert(OFF % 16 == 0 && OFF >= -2048 && OFF <= 2032, "a quad's offset is a multiple of 16");
  __VX_WIDE_Q(__VX_EMIT_FSW_Q, _)
}

#undef __VX_FLW
#undef __VX_FLW_
#undef __VX_FLW4
#undef __VX_FLW4_
#undef __VX_FSW
#undef __VX_FSW_
#undef __VX_FSW4
#undef __VX_FSW4_
#undef __VX_WIDE_Q
#undef __VX_WIDE_D
#undef __VX_EMIT_FLW_D
#undef __VX_EMIT_FLW_Q
#undef __VX_EMIT_FSW_D
#undef __VX_EMIT_FSW_Q

#endif // VX_CFG_EXT_LDST_WIDE_ENABLE

//
// Matrix loads and stores
//

// vx_ldm fd, imm(rs1) / vx_stm fs2, imm(rs1) shapes (imm[4:0]): a warp's matrix
// fragment of NR registers in the fragment register layout, the A/B/C operand
// layouts of the warp MMA; imm[11:5] is a byte offset.
enum vx_ldm_shape_t : int32_t {
  VX_LDM_A    = 0x00, // A, row-major
  VX_LDM_A_SP = 0x01, // sparse A, row-major
  VX_LDM_B    = 0x06, // B, col-major
  VX_LDM_B_SP = 0x07, // symmetric sparse B, col-major
  VX_LDM_C    = 0x08, // C, row-major
  VX_LDM_WG_C = 0x18  // warp-group C, row-major, one 8-register chunk
};

#ifdef VX_CFG_EXT_LDST_MATRIX_ENABLE

// vx_ldm/vx_stm of NR registers from fd/fs2 = FD with immediate IMM (the shape
// ORed with a byte offset). g holds the lane's address of each of the J row
// groups: J = 2 rides in an even-aligned register pair, J = 4 is pinned to
// x[Q]..x[Q+3].
#if __riscv_xlen == 64
typedef unsigned __int128 __vx_gpr_pair_t;
#else
typedef uint64_t __vx_gpr_pair_t;
#endif

#define __VX_LDM8(...) __VX_LDM8_(__VA_ARGS__)
#define __VX_LDM8_(r0, r1, r2, r3, r4, r5, r6, r7, ...)                                  \
    register float x0 __asm__(r0); register float x1 __asm__(r1);                         \
    register float x2 __asm__(r2); register float x3 __asm__(r3);                         \
    register float x4 __asm__(r4); register float x5 __asm__(r5);                         \
    register float x6 __asm__(r6); register float x7 __asm__(r7);                         \
    __asm__ volatile (".insn i %[op], 0, %[x0], %[g], %[imm]"                             \
      : [x0]"=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3), "=f"(x4), "=f"(x5), "=f"(x6), "=f"(x7)\
      : [op]"i"(RISCV_CUSTOM3), [imm]"i"(IMM), __VA_ARGS__ : "memory");                   \
    v[0] = x0; v[1] = x1; v[2] = x2; v[3] = x3; v[4] = x4; v[5] = x5; v[6] = x6; v[7] = x7;
#define __VX_LDM4(...) __VX_LDM4_(__VA_ARGS__)
#define __VX_LDM4_(r0, r1, r2, r3, ...)                                                  \
    register float x0 __asm__(r0); register float x1 __asm__(r1);                         \
    register float x2 __asm__(r2); register float x3 __asm__(r3);                         \
    __asm__ volatile (".insn i %[op], 0, %[x0], %[g], %[imm]"                             \
      : [x0]"=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)                                        \
      : [op]"i"(RISCV_CUSTOM3), [imm]"i"(IMM), __VA_ARGS__ : "memory");                   \
    v[0] = x0; v[1] = x1; v[2] = x2; v[3] = x3;
#define __VX_STM8(...) __VX_STM8_(__VA_ARGS__)
#define __VX_STM8_(r0, r1, r2, r3, r4, r5, r6, r7, ...)                                  \
    register float x0 __asm__(r0) = v[0]; register float x1 __asm__(r1) = v[1];           \
    register float x2 __asm__(r2) = v[2]; register float x3 __asm__(r3) = v[3];           \
    register float x4 __asm__(r4) = v[4]; register float x5 __asm__(r5) = v[5];           \
    register float x6 __asm__(r6) = v[6]; register float x7 __asm__(r7) = v[7];           \
    __asm__ volatile (".insn s %[op], 2, %[x0], %[imm](%[g])"                             \
      :: [op]"i"(RISCV_CUSTOM3), [imm]"i"(IMM), [x0]"f"(x0), "f"(x1), "f"(x2), "f"(x3),    \
         "f"(x4), "f"(x5), "f"(x6), "f"(x7), __VA_ARGS__ : "memory");
#define __VX_STM4(...) __VX_STM4_(__VA_ARGS__)
#define __VX_STM4_(r0, r1, r2, r3, ...)                                                  \
    register float x0 __asm__(r0) = v[0]; register float x1 __asm__(r1) = v[1];           \
    register float x2 __asm__(r2) = v[2]; register float x3 __asm__(r3) = v[3];           \
    __asm__ volatile (".insn s %[op], 2, %[x0], %[imm](%[g])"                             \
      :: [op]"i"(RISCV_CUSTOM3), [imm]"i"(IMM), [x0]"f"(x0), "f"(x1), "f"(x2), "f"(x3),    \
         __VA_ARGS__ : "memory");

#define __VX_MTX8(EMIT, ...) \
  if constexpr (FD == 0) { EMIT(__VX_FREGS8(0,1,2,3,4,5,6,7), __VA_ARGS__) } else \
  if constexpr (FD == 1) { EMIT(__VX_FREGS8(1,2,3,4,5,6,7,8), __VA_ARGS__) } else \
  if constexpr (FD == 2) { EMIT(__VX_FREGS8(2,3,4,5,6,7,8,9), __VA_ARGS__) } else \
  if constexpr (FD == 3) { EMIT(__VX_FREGS8(3,4,5,6,7,8,9,10), __VA_ARGS__) } else \
  if constexpr (FD == 4) { EMIT(__VX_FREGS8(4,5,6,7,8,9,10,11), __VA_ARGS__) } else \
  if constexpr (FD == 5) { EMIT(__VX_FREGS8(5,6,7,8,9,10,11,12), __VA_ARGS__) } else \
  if constexpr (FD == 6) { EMIT(__VX_FREGS8(6,7,8,9,10,11,12,13), __VA_ARGS__) } else \
  if constexpr (FD == 7) { EMIT(__VX_FREGS8(7,8,9,10,11,12,13,14), __VA_ARGS__) } else \
  if constexpr (FD == 8) { EMIT(__VX_FREGS8(8,9,10,11,12,13,14,15), __VA_ARGS__) } else \
  if constexpr (FD == 9) { EMIT(__VX_FREGS8(9,10,11,12,13,14,15,16), __VA_ARGS__) } else \
  if constexpr (FD == 10) { EMIT(__VX_FREGS8(10,11,12,13,14,15,16,17), __VA_ARGS__) } else \
  if constexpr (FD == 11) { EMIT(__VX_FREGS8(11,12,13,14,15,16,17,18), __VA_ARGS__) } else \
  if constexpr (FD == 12) { EMIT(__VX_FREGS8(12,13,14,15,16,17,18,19), __VA_ARGS__) } else \
  if constexpr (FD == 13) { EMIT(__VX_FREGS8(13,14,15,16,17,18,19,20), __VA_ARGS__) } else \
  if constexpr (FD == 14) { EMIT(__VX_FREGS8(14,15,16,17,18,19,20,21), __VA_ARGS__) } else \
  if constexpr (FD == 15) { EMIT(__VX_FREGS8(15,16,17,18,19,20,21,22), __VA_ARGS__) } else \
  if constexpr (FD == 16) { EMIT(__VX_FREGS8(16,17,18,19,20,21,22,23), __VA_ARGS__) } else \
  if constexpr (FD == 17) { EMIT(__VX_FREGS8(17,18,19,20,21,22,23,24), __VA_ARGS__) } else \
  if constexpr (FD == 18) { EMIT(__VX_FREGS8(18,19,20,21,22,23,24,25), __VA_ARGS__) } else \
  if constexpr (FD == 19) { EMIT(__VX_FREGS8(19,20,21,22,23,24,25,26), __VA_ARGS__) } else \
  if constexpr (FD == 20) { EMIT(__VX_FREGS8(20,21,22,23,24,25,26,27), __VA_ARGS__) } else \
  if constexpr (FD == 21) { EMIT(__VX_FREGS8(21,22,23,24,25,26,27,28), __VA_ARGS__) } else \
  if constexpr (FD == 22) { EMIT(__VX_FREGS8(22,23,24,25,26,27,28,29), __VA_ARGS__) } else \
  if constexpr (FD == 23) { EMIT(__VX_FREGS8(23,24,25,26,27,28,29,30), __VA_ARGS__) } else \
  if constexpr (FD == 24) { EMIT(__VX_FREGS8(24,25,26,27,28,29,30,31), __VA_ARGS__) } else \
  { static_assert(FD + 8 <= 32, "an 8-register fragment's fd is at most 24"); }

#define __VX_MTX4(EMIT, ...) \
  if constexpr (FD == 0) { EMIT(__VX_FREGS4(0,1,2,3), __VA_ARGS__) } else \
  if constexpr (FD == 1) { EMIT(__VX_FREGS4(1,2,3,4), __VA_ARGS__) } else \
  if constexpr (FD == 2) { EMIT(__VX_FREGS4(2,3,4,5), __VA_ARGS__) } else \
  if constexpr (FD == 3) { EMIT(__VX_FREGS4(3,4,5,6), __VA_ARGS__) } else \
  if constexpr (FD == 4) { EMIT(__VX_FREGS4(4,5,6,7), __VA_ARGS__) } else \
  if constexpr (FD == 5) { EMIT(__VX_FREGS4(5,6,7,8), __VA_ARGS__) } else \
  if constexpr (FD == 6) { EMIT(__VX_FREGS4(6,7,8,9), __VA_ARGS__) } else \
  if constexpr (FD == 7) { EMIT(__VX_FREGS4(7,8,9,10), __VA_ARGS__) } else \
  if constexpr (FD == 8) { EMIT(__VX_FREGS4(8,9,10,11), __VA_ARGS__) } else \
  if constexpr (FD == 9) { EMIT(__VX_FREGS4(9,10,11,12), __VA_ARGS__) } else \
  if constexpr (FD == 10) { EMIT(__VX_FREGS4(10,11,12,13), __VA_ARGS__) } else \
  if constexpr (FD == 11) { EMIT(__VX_FREGS4(11,12,13,14), __VA_ARGS__) } else \
  if constexpr (FD == 12) { EMIT(__VX_FREGS4(12,13,14,15), __VA_ARGS__) } else \
  if constexpr (FD == 13) { EMIT(__VX_FREGS4(13,14,15,16), __VA_ARGS__) } else \
  if constexpr (FD == 14) { EMIT(__VX_FREGS4(14,15,16,17), __VA_ARGS__) } else \
  if constexpr (FD == 15) { EMIT(__VX_FREGS4(15,16,17,18), __VA_ARGS__) } else \
  if constexpr (FD == 16) { EMIT(__VX_FREGS4(16,17,18,19), __VA_ARGS__) } else \
  if constexpr (FD == 17) { EMIT(__VX_FREGS4(17,18,19,20), __VA_ARGS__) } else \
  if constexpr (FD == 18) { EMIT(__VX_FREGS4(18,19,20,21), __VA_ARGS__) } else \
  if constexpr (FD == 19) { EMIT(__VX_FREGS4(19,20,21,22), __VA_ARGS__) } else \
  if constexpr (FD == 20) { EMIT(__VX_FREGS4(20,21,22,23), __VA_ARGS__) } else \
  if constexpr (FD == 21) { EMIT(__VX_FREGS4(21,22,23,24), __VA_ARGS__) } else \
  if constexpr (FD == 22) { EMIT(__VX_FREGS4(22,23,24,25), __VA_ARGS__) } else \
  if constexpr (FD == 23) { EMIT(__VX_FREGS4(23,24,25,26), __VA_ARGS__) } else \
  if constexpr (FD == 24) { EMIT(__VX_FREGS4(24,25,26,27), __VA_ARGS__) } else \
  if constexpr (FD == 25) { EMIT(__VX_FREGS4(25,26,27,28), __VA_ARGS__) } else \
  if constexpr (FD == 26) { EMIT(__VX_FREGS4(26,27,28,29), __VA_ARGS__) } else \
  if constexpr (FD == 27) { EMIT(__VX_FREGS4(27,28,29,30), __VA_ARGS__) } else \
  if constexpr (FD == 28) { EMIT(__VX_FREGS4(28,29,30,31), __VA_ARGS__) } else \
  { static_assert(FD + 4 <= 32, "a 4-register fragment's fd is at most 28"); }

#define __VX_MTX(EMIT8, EMIT4, ...)                                                      \
    if constexpr (NR == 8) { __VX_MTX8(EMIT8, __VA_ARGS__) }                              \
    else { static_assert(NR == 4, "a fragment has 4 or 8 registers");                     \
           __VX_MTX4(EMIT4, __VA_ARGS__) }

#define __VX_GQ(q) "x" #q
#define __VX_MTX_GROUP(EMIT8, EMIT4)                                                     \
    static_assert(J == 2 || J == 4, "a fragment has 2 or 4 row groups");                  \
    if constexpr (J == 2) {                                                               \
      __vx_gpr_pair_t p = __vx_gpr_pair_t(g[0]) | (__vx_gpr_pair_t(g[1]) << (8 * sizeof(uintptr_t))); \
      __VX_MTX(EMIT8, EMIT4, [g]"R"(p))                                                   \
    }                                                                                     \
    __VX_MTX_GROUP4(EMIT8, EMIT4, 12, 13, 14, 15)                                         \
    __VX_MTX_GROUP4(EMIT8, EMIT4, 16, 17, 18, 19)                                         \
    __VX_MTX_GROUP4(EMIT8, EMIT4, 20, 21, 22, 23)                                         \
    __VX_MTX_GROUP4(EMIT8, EMIT4, 24, 25, 26, 27)                                         \
    __VX_MTX_GROUP4(EMIT8, EMIT4, 28, 29, 30, 31)                                         \
    else { static_assert(J == 2, "a 4-register group starts at x12, x16, x20, x24 or x28"); }
#define __VX_MTX_GROUP4(EMIT8, EMIT4, a, b, c, d)                                        \
    else if constexpr (Q == a) {                                                          \
      register uintptr_t q0 __asm__(__VX_GQ(a)) = g[0]; register uintptr_t q1 __asm__(__VX_GQ(b)) = g[1]; \
      register uintptr_t q2 __asm__(__VX_GQ(c)) = g[2]; register uintptr_t q3 __asm__(__VX_GQ(d)) = g[3]; \
      __VX_MTX(EMIT8, EMIT4, [g]"r"(q0), "r"(q1), "r"(q2), "r"(q3))                       \
    }

template <uint32_t FD, uint32_t NR, uint32_t J, uint32_t Q, int32_t IMM>
__attribute__((always_inline)) inline void vx_ldm(float (&v)[NR], const uintptr_t (&g)[J]) {
  __VX_MTX_GROUP(__VX_LDM8, __VX_LDM4)
}

template <uint32_t FD, uint32_t NR, uint32_t J, uint32_t Q, int32_t IMM>
__attribute__((always_inline)) inline void vx_stm(const float (&v)[NR], const uintptr_t (&g)[J]) {
  __VX_MTX_GROUP(__VX_STM8, __VX_STM4)
}

#undef __VX_LDM8
#undef __VX_LDM8_
#undef __VX_LDM4
#undef __VX_LDM4_
#undef __VX_STM8
#undef __VX_STM8_
#undef __VX_STM4
#undef __VX_STM4_
#undef __VX_MTX8
#undef __VX_MTX4
#undef __VX_MTX
#undef __VX_GQ
#undef __VX_MTX_GROUP
#undef __VX_MTX_GROUP4

#endif // VX_CFG_EXT_LDST_MATRIX_ENABLE

#if defined(VX_CFG_EXT_LDST_WIDE_ENABLE) || defined(VX_CFG_EXT_LDST_MATRIX_ENABLE)
#undef __VX_FREGS2
#undef __VX_FREGS4
#undef __VX_FREGS8
#endif

#endif // __cplusplus

#endif // __VX_INTRINSICS_H__
