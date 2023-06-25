#ifndef __VX_INTRINSICS_H__
#define __VX_INTRINSICS_H__

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
    unsigned __r;                                               \
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
    unsigned __r;                                               \
    __asm__ __volatile__ (                                      \
        ".word ((" __ASM_STR(opcode) ") | (reg_%0 << 7) | (" __ASM_STR(imm1) " << 15) | (" __ASM_STR(imm2) " << 20) | ((" __ASM_STR(func3) ") << 12) | ((" __ASM_STR(func7) ") << 25));" \
        : "=r" (__r) :                                          \
    );                                                          \
    __r;                                                        \
})

#define RISCV_INSN_R4(opcode, func3, func2, rs1, rs2, rs3) ({   \
    unsigned __r;                                               \
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
	unsigned __r;	               		        \
	__asm__ __volatile__ ("csrr %0, %1" : "=r" (__r) : "i" (csr)); \
	__r;							            \
})

#define csr_write(csr, val)	({                  \
	unsigned __v = (unsigned)(val);             \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ __volatile__ ("csrw %0, %1"	:: "i" (csr), "i" (__v));  \
    else                                        \
        __asm__ __volatile__ ("csrw %0, %1"	:: "i" (csr), "r" (__v));  \
})

#define csr_swap(csr, val) ({                   \
    unsigned __r;                               \
	unsigned __v = (unsigned)(val);	            \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ __volatile__ ("csrrw %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ __volatile__ ("csrrw %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v)); \
	__r;						                \
})

#define csr_read_set(csr, val) ({               \
	unsigned __r;                               \
	unsigned __v = (unsigned)(val);	            \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ __volatile__ ("csrrs %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ __volatile__ ("csrrs %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v)); \
	__r;							            \
})

#define csr_set(csr, val) ({                    \
	unsigned __v = (unsigned)(val);	            \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ __volatile__ ("csrs %0, %1"	:: "i" (csr), "i" (__v));  \
    else                                        \
        __asm__ __volatile__ ("csrs %0, %1"	:: "i" (csr), "r" (__v));  \
})

#define csr_read_clear(csr, val) ({             \
	unsigned __r;                               \
	unsigned __v = (unsigned)(val);	            \
    if (__builtin_constant_p(val) && __v < 32)  \
	    __asm__ __volatile__ ("csrrc %0, %1, %2" : "=r" (__r) : "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ __volatile__ ("csrrc %0, %1, %2" : "=r" (__r) : "i" (csr), "r" (__v)); \
	__r;							            \
})

#define csr_clear(csr, val)	({                  \
	unsigned __v = (unsigned)(val);             \
	if (__builtin_constant_p(val) && __v < 32)  \
        __asm__ __volatile__ ("csrc %0, %1"	:: "i" (csr), "i" (__v)); \
    else                                        \
        __asm__ __volatile__ ("csrc %0, %1"	:: "i" (csr), "r" (__v)); \
})

// Texture load
#define vx_tex(stage, u, v, lod) \
    RISCV_INSN_R4(RISCV_CUSTOM1, 0, stage, u, v, lod)

// Conditional move
inline unsigned vx_cmov(unsigned c, unsigned t, unsigned f) {
	return RISCV_INSN_R4(RISCV_CUSTOM1, 1, 0, c, t, f);
}

// Rop write
inline void vx_rop(unsigned x, unsigned y, unsigned face, unsigned color, unsigned depth) {
    unsigned pos_face = (y << 16) | (x << 1) | face;
    RISCV_INSN_R4_0111(RISCV_CUSTOM1, 1, 1, pos_face, color, depth);
}

// Integer multiply add
#define vx_imadd(a, b, c, shift) \
    RISCV_INSN_R4(RISCV_CUSTOM1, 2, shift, a, b, c)

// Raster load
inline unsigned vx_rast() {
    return RISCV_INSN_R_100(RISCV_CUSTOM0, 0, 1, 0, 0);
}

// Set thread mask
inline void vx_tmc(unsigned thread_mask) {
    RISCV_INSN_R_010(RISCV_CUSTOM0, 0, 0, thread_mask, 0);
}

// Set thread predicate
inline void vx_pred(unsigned condition) {
    RISCV_INSN_R_010(RISCV_CUSTOM0, 0, 0, condition, 1);
}

typedef void (*vx_wspawn_pfn)();

// Spawn warps
inline void vx_wspawn(unsigned num_warps, vx_wspawn_pfn func_ptr) {
    RISCV_INSN_R_011(RISCV_CUSTOM0, 1, 0, num_warps, func_ptr);
}

// Split on a predicate
inline void vx_split(unsigned predicate) {
    RISCV_INSN_R_010(RISCV_CUSTOM0, 2, 0, predicate, 0);
}

// Join
inline void vx_join() {
  RISCV_INSN_R_000(RISCV_CUSTOM0, 3, 0, 0, 0);
}

// Warp Barrier
inline void vx_barrier(unsigned barried_id, unsigned num_warps) {
    RISCV_INSN_R_011(RISCV_CUSTOM0, 4, 0, barried_id, num_warps);
}

// Return current thread identifier
inline int vx_thread_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_THREAD_ID));
    return result;
}

// Return current warp identifier
inline int vx_warp_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_WARP_ID));
    return result;
}

// Return current core identifier
inline int vx_core_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_CORE_ID));
    return result;
}

// Return current cluster identifier
inline int vx_cluster_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_CLUSTER_ID));
    return result;
}

// Return current threadk mask
inline int vx_thread_mask() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_TMASK));
    return result;
}

// Return the number of threads per warp
inline int vx_num_threads() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_NUM_THREADS));
    return result;
}

// Return the number of warps per core
inline int vx_num_warps() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_NUM_WARPS));
    return result;   
}

// Return the number of cores per cluster
inline int vx_num_cores() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_NUM_CORES));
    return result;
}

// Return the number of clusters
inline int vx_num_clusters() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_NUM_CLUSTERS));
    return result;
}

// Return the hart identifier (thread id accross the processor)
inline int vx_hart_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(VX_CSR_MHARTID));
    return result;
}

inline void vx_fence() {
    asm volatile ("fence iorw, iorw");
}

#ifdef __cplusplus
}
#endif

#endif // __VX_INTRINSICS_H__
