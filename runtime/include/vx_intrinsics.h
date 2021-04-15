#ifndef VX_INTRINSICS_H
#define VX_INTRINSICS_H

#include <VX_config.h>

#ifdef __cplusplus
extern "C" {
#endif

// Set thread mask
inline void vx_tmc(unsigned num_threads) {
    asm volatile (".insn s 0x6b, 0, x0, 0(%0)" :: "r"(num_threads));
}

// Spawn warps
inline void vx_wspawn(unsigned num_warps, void* func_ptr) {
    asm volatile (".insn s 0x6b, 1, %1, 0(%0)" :: "r"(num_warps), "r"(func_ptr));
}

// Split on a predicate
inline void vx_split(int predicate) {
    asm volatile (".insn s 0x6b, 2, x0, 0(%0)" :: "r"(predicate));
}

// Join
inline void vx_join() {
  asm volatile (".insn s 0x6b, 3, x0, 0(x0)");
}

// Warp Barrier
inline void vx_barrier(unsigned barried_id, unsigned num_warps) {
    asm volatile (".insn s 0x6b, 4, %1, 0cd (%0)" :: "r"(barried_id), "r"(num_warps));
}

// Return active warp's thread id 
inline int vx_thread_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_WTID));
    return result;   
}

// Return active core's local thread id
inline int vx_thread_lid() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_LTID));
    return result;   
}

// Return processsor global thread id
inline int vx_thread_gid() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_GTID));
    return result;   
}

// Return active core's local warp id
inline int vx_warp_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_LWID));
    return result;   
}

// Return processsor's global warp id
inline int vx_warp_gid() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_GWID));
    return result;   
}

// Return processsor core id
inline int vx_core_id() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_GCID));
    return result; 
}

// Return the number of threads in a warp
inline int vx_num_threads() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_NT));
    return result; 
}

// Return the number of warps in a core
inline int vx_num_warps() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_NW));
    return result;   
}

// Return the number of cores in the processsor
inline int vx_num_cores() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_NC));
    return result;   
}

// Return the number of cycles
inline int vx_num_cycles() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_CYCLE));
    return result;   
}

// Return the number of instructions
inline int vx_num_instrs() {
    int result;
    asm volatile ("csrr %0, %1" : "=r"(result) : "i"(CSR_INSTRET));
    return result; 
}

// TODO: delete all 3 of me
inline uint32_t __intrin_add(uint32_t a, uint32_t b) {
    uint32_t ret;
    asm volatile (
        "add %[rd], %[rs1], %[rs2]"
        : [rd] "=r" (ret)
        : [rs1] "r" (a), [rs2] "r" (b));
    return ret;
}

inline void __intrin_add_cursed(uint32_t *arr) {
    asm volatile (
        "add %[rd], %[rs1], %[rs2]"
        : [rd] "=r" (arr[2])
        : [rs1] "r" (arr[0]), [rs2] "r" (arr[1]));
}

inline void __intrin_add_more_cursed(uint32_t *arr) {
    asm volatile (
        "add %[rd], %[rs1], %[rd]"
        : [rd] "+r" (arr[1])
        : [rs1] "r" (arr[0]));
}

//
// SHA-256
//
inline uint32_t __intrin_sha_sigma0(uint32_t x) {
    uint32_t ret;
    asm volatile (
        ".insn i 0x13, 1, %[rd], %[rs1], 0x102\n"
        : [rd] "=r" (ret)
        : [rs1] "r" (x));
    return ret;
}

inline uint32_t __intrin_sha_sigma1(uint32_t x) {
    uint32_t ret;
    asm volatile (
        ".insn i 0x13, 1, %[rd], %[rs1], 0x103\n"
        : [rd] "=r" (ret)
        : [rs1] "r" (x));
    return ret;
}

inline uint32_t __intrin_sha_Sigma0(uint32_t x) {
    uint32_t ret;
    asm volatile (
        ".insn i 0x13, 1, %[rd], %[rs1], 0x100\n"
        : [rd] "=r" (ret)
        : [rs1] "r" (x));
    return ret;
}

inline uint32_t __intrin_sha_Sigma1(uint32_t x) {
    uint32_t ret;
    asm volatile (
        ".insn i 0x13, 1, %[rd], %[rs1], 0x101\n"
        : [rd] "=r" (ret)
        : [rs1] "r" (x));
    return ret;
}

//
// AES-256
//
inline void __intrin_aes_enc_round(uint32_t *newcols, uint32_t *oldcols) {
    // aes32esmi
    asm volatile (
        // See:
        // https://sourceware.org/binutils/docs-2.36/as/RISC_002dV_002dFormats.html
        ".insn r 0x33, 0, 0x1b, x0, %[n0], %[o0]\n"
        ".insn r 0x33, 0, 0x3b, x0, %[n0], %[o1]\n"
        ".insn r 0x33, 0, 0x5b, x0, %[n0], %[o2]\n"
        ".insn r 0x33, 0, 0x7b, x0, %[n0], %[o3]\n"
        ".insn r 0x33, 0, 0x1b, x0, %[n1], %[o1]\n"
        ".insn r 0x33, 0, 0x3b, x0, %[n1], %[o2]\n"
        ".insn r 0x33, 0, 0x5b, x0, %[n1], %[o3]\n"
        ".insn r 0x33, 0, 0x7b, x0, %[n1], %[o0]\n"
        ".insn r 0x33, 0, 0x1b, x0, %[n2], %[o2]\n"
        ".insn r 0x33, 0, 0x3b, x0, %[n2], %[o3]\n"
        ".insn r 0x33, 0, 0x5b, x0, %[n2], %[o0]\n"
        ".insn r 0x33, 0, 0x7b, x0, %[n2], %[o1]\n"
        ".insn r 0x33, 0, 0x1b, x0, %[n3], %[o3]\n"
        ".insn r 0x33, 0, 0x3b, x0, %[n3], %[o0]\n"
        ".insn r 0x33, 0, 0x5b, x0, %[n3], %[o1]\n"
        ".insn r 0x33, 0, 0x7b, x0, %[n3], %[o2]"
        : [n0] "+r" (newcols[0]), [n1] "+r" (newcols[1]),
          [n2] "+r" (newcols[2]), [n3] "+r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]));
}

inline void __intrin_aes_last_enc_round(uint32_t *newcols, uint32_t *oldcols) {
    // aes32esi
    asm volatile (
        // See:
        // https://sourceware.org/binutils/docs-2.36/as/RISC_002dV_002dFormats.html
        ".insn r 0x33, 0, 0x19, x0, %[n0], %[o0]\n"
        ".insn r 0x33, 0, 0x39, x0, %[n0], %[o1]\n"
        ".insn r 0x33, 0, 0x59, x0, %[n0], %[o2]\n"
        ".insn r 0x33, 0, 0x79, x0, %[n0], %[o3]\n"
        ".insn r 0x33, 0, 0x19, x0, %[n1], %[o1]\n"
        ".insn r 0x33, 0, 0x39, x0, %[n1], %[o2]\n"
        ".insn r 0x33, 0, 0x59, x0, %[n1], %[o3]\n"
        ".insn r 0x33, 0, 0x79, x0, %[n1], %[o0]\n"
        ".insn r 0x33, 0, 0x19, x0, %[n2], %[o2]\n"
        ".insn r 0x33, 0, 0x39, x0, %[n2], %[o3]\n"
        ".insn r 0x33, 0, 0x59, x0, %[n2], %[o0]\n"
        ".insn r 0x33, 0, 0x79, x0, %[n2], %[o1]\n"
        ".insn r 0x33, 0, 0x19, x0, %[n3], %[o3]\n"
        ".insn r 0x33, 0, 0x39, x0, %[n3], %[o0]\n"
        ".insn r 0x33, 0, 0x59, x0, %[n3], %[o1]\n"
        ".insn r 0x33, 0, 0x79, x0, %[n3], %[o2]"
        : [n0] "+r" (newcols[0]), [n1] "+r" (newcols[1]),
          [n2] "+r" (newcols[2]), [n3] "+r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]));
}

inline void __intrin_aes_dec_round(uint32_t *newcols, uint32_t *oldcols) {
    // aes32dsmi
    asm volatile (
        // See:
        // https://sourceware.org/binutils/docs-2.36/as/RISC_002dV_002dFormats.html
        ".insn r 0x33, 0, 0x1f, x0, %[n0], %[o0]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n0], %[o3]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n0], %[o2]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n0], %[o1]\n"
        ".insn r 0x33, 0, 0x1f, x0, %[n1], %[o1]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n1], %[o0]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n1], %[o3]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n1], %[o2]\n"
        ".insn r 0x33, 0, 0x1f, x0, %[n2], %[o2]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n2], %[o1]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n2], %[o0]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n2], %[o3]\n"
        ".insn r 0x33, 0, 0x1f, x0, %[n3], %[o3]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n3], %[o2]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n3], %[o1]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n3], %[o0]"
        : [n0] "+r" (newcols[0]), [n1] "+r" (newcols[1]),
          [n2] "+r" (newcols[2]), [n3] "+r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]));
}

inline void __intrin_aes_last_dec_round(uint32_t *newcols, uint32_t *oldcols) {
    // aes32dsi
    asm volatile (
        // See:
        // https://sourceware.org/binutils/docs-2.36/as/RISC_002dV_002dFormats.html
        ".insn r 0x33, 0, 0x1d, x0, %[n0], %[o0]\n"
        ".insn r 0x33, 0, 0x3d, x0, %[n0], %[o3]\n"
        ".insn r 0x33, 0, 0x5d, x0, %[n0], %[o2]\n"
        ".insn r 0x33, 0, 0x7d, x0, %[n0], %[o1]\n"
        ".insn r 0x33, 0, 0x1d, x0, %[n1], %[o1]\n"
        ".insn r 0x33, 0, 0x3d, x0, %[n1], %[o0]\n"
        ".insn r 0x33, 0, 0x5d, x0, %[n1], %[o3]\n"
        ".insn r 0x33, 0, 0x7d, x0, %[n1], %[o2]\n"
        ".insn r 0x33, 0, 0x1d, x0, %[n2], %[o2]\n"
        ".insn r 0x33, 0, 0x3d, x0, %[n2], %[o1]\n"
        ".insn r 0x33, 0, 0x5d, x0, %[n2], %[o0]\n"
        ".insn r 0x33, 0, 0x7d, x0, %[n2], %[o3]\n"
        ".insn r 0x33, 0, 0x1d, x0, %[n3], %[o3]\n"
        ".insn r 0x33, 0, 0x3d, x0, %[n3], %[o2]\n"
        ".insn r 0x33, 0, 0x5d, x0, %[n3], %[o1]\n"
        ".insn r 0x33, 0, 0x7d, x0, %[n3], %[o0]"
        : [n0] "+r" (newcols[0]), [n1] "+r" (newcols[1]),
          [n2] "+r" (newcols[2]), [n3] "+r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]));
}

#define __if(b) vx_split(b); \
                if (b) 

#define __else else

#define __endif vx_join();

#ifdef __cplusplus
}
#endif

#endif
