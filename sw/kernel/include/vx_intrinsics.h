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

// Set thread mask
inline void vx_tmc(int thread_mask) {
    __asm__ volatile (".insn r %0, 0, 0, x0, %1, x0" :: "i"(RISCV_CUSTOM0), "r"(thread_mask) : "memory");
}

// disable all threads in the current warp
inline void vx_tmc_zero() {
    __asm__ volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM0) : "memory");
}

// switch execution to single thread0
inline void vx_tmc_one() {
    __asm__ volatile (
        "li a0, 1\n\t"  // Load immediate value 1 into a0 (x10) register
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

// Return current thread identifier
inline __attribute__((const)) int vx_thread_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_THREAD_ID));
    return ret;
}

// Return current warp identifier
inline __attribute__((const)) int vx_warp_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_WARP_ID));
    return ret;
}

// Return current core identifier
inline __attribute__((const)) int vx_core_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_CORE_ID));
    return ret;
}

// Return active threads mask
inline __attribute__((const)) size_t vx_active_threads() {
    size_t ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_ACTIVE_THREADS));
    return ret;
}

// Return active warps mask
inline __attribute__((const)) size_t vx_active_warps() {
    size_t ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_ACTIVE_WARPS));
    return ret;
}

// Return the number of threads per warp
inline __attribute__((const)) int vx_num_threads() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_THREADS));
    return ret;
}

// Return the number of warps per core
inline __attribute__((const)) int vx_num_warps() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_WARPS));
    return ret;
}

// Return the number of cores per cluster
inline __attribute__((const)) int vx_num_cores() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_CORES));
    return ret;
}

// Return the number of barriers
inline __attribute__((const)) int vx_num_barriers() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_NUM_BARRIERS));
    return ret;
}

// Return the hart identifier (thread id accross the processor)
inline __attribute__((const)) int vx_hart_id() {
    int ret;
    __asm__ volatile ("csrr %0, %1" : "=r"(ret) : "i"(VX_CSR_MHARTID));
    return ret;
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

// Graphics extensions (CUSTOM1 / INST_EXT2 family).
// Encodings:
//   funct3=1, R4-type, funct2=stage : vx_tex   (texture sample)
//   funct3=2, R4-type, funct2=0     : vx_om    (output-merger write)
//   funct3=3, R-type,  funct7=0     : vx_rast  (raster pop)
// These trap as illegal-instruction unless the corresponding
// EXT_TEX_ENABLE / EXT_OM_ENABLE / EXT_RASTER_ENABLE is set.

// Texture sample: (stage, u, v, lod) -> texel
inline unsigned vx_tex(unsigned stage, unsigned u, unsigned v, unsigned lod) {
    unsigned ret;
    __asm__ volatile (".insn r4 %1, 1, %2, %0, %3, %4, %5"
        : "=r"(ret)
        : "i"(RISCV_CUSTOM1), "i"(stage), "r"(u), "r"(v), "r"(lod));
    return ret;
}

// Output-merger write: (x, y, face, color, depth)
inline void vx_om(unsigned x, unsigned y, unsigned face, unsigned color, unsigned depth) {
    unsigned pos_face = (y << 16) | (x << 1) | face;
    __asm__ volatile (".insn r4 %0, 2, 0, x0, %1, %2, %3"
        :: "i"(RISCV_CUSTOM1), "r"(pos_face), "r"(color), "r"(depth));
}

// Raster pop: returns next quad descriptor from the rasterizer.
inline unsigned vx_rast() {
    unsigned ret;
    __asm__ volatile (".insn r %1, 3, 0, %0, x0, x0"
        : "=r"(ret) : "i"(RISCV_CUSTOM1));
    return ret;
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
    __asm__ volatile ("fence iorw, iorw");
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
// Warp-Level Lane Gather Extension
//

// Each lane gathers a value from the source lane's register file.
// S = source lane (compile-time constant, 0-3).
// The source lane retains its own rd value; lane (S+1) gets v1[S], (S+2) gets v2[S], (S+3) gets v3[S].
#define __VX_WGATHER(src_lane, self_val, v1, v2, v3) ({ \
    size_t __ret = (self_val);                          \
    __asm__ volatile (                                  \
        ".insn r4 %1, 0, %2, %0, %3, %4, %5"            \
        : "+r"(__ret)                                   \
        : "i"(RISCV_CUSTOM1), "i"(src_lane),            \
          "r"(v1), "r"(v2), "r"(v3)                     \
    );                                                  \
    __ret;                                              \
})

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
inline float vx_packlb_f(const void* base, uint32_t stride) {
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
inline float vx_packlh_f(const void* base, uint32_t stride) {
    float out;
    __asm__ volatile (
        ".insn r %1, 2, 4, %0, %2, %3" : "=f"(out) : "i"(RISCV_CUSTOM0), "r"(base), "r"(stride) : "memory"
    );
    return out;
}

#ifdef __cplusplus
}
#endif

#endif // __VX_INTRINSICS_H__
