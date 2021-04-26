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
inline void __intrin_aes_enc_round(uint32_t *newcols,
                                   const uint32_t *oldcols,
                                   const uint32_t *round_key) {
    // aes32esmi
    asm volatile (
        "mv %[n0], %[k0]\n"
        "mv %[n1], %[k1]\n"
        "mv %[n2], %[k2]\n"
        "mv %[n3], %[k3]\n"
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
        : [n0] "=&r" (newcols[0]), [n1] "=&r" (newcols[1]),
          [n2] "=&r" (newcols[2]), [n3] "=&r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]),
          [k0] "r" (round_key[0]), [k1] "r" (round_key[1]),
          [k2] "r" (round_key[2]), [k3] "r" (round_key[3]));
}

inline void __intrin_aes_last_enc_round(uint32_t *newcols,
                                        const uint32_t *oldcols,
                                        const uint32_t *round_key) {
    // aes32esi
    asm volatile (
        "mv %[n0], %[k0]\n"
        "mv %[n1], %[k1]\n"
        "mv %[n2], %[k2]\n"
        "mv %[n3], %[k3]\n"
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
        : [n0] "=&r" (newcols[0]), [n1] "=&r" (newcols[1]),
          [n2] "=&r" (newcols[2]), [n3] "=&r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]),
          [k0] "r" (round_key[0]), [k1] "r" (round_key[1]),
          [k2] "r" (round_key[2]), [k3] "r" (round_key[3]));
}

inline void __intrin_aes_dec_round(uint32_t *newcols, const uint32_t *oldcols,
                                   const uint32_t *round_key) {
    // aes32dsmi
    asm volatile (
        "mv %[n0], %[k0]\n"
        "mv %[n1], %[k1]\n"
        "mv %[n2], %[k2]\n"
        "mv %[n3], %[k3]\n"
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
        : [n0] "=&r" (newcols[0]), [n1] "=&r" (newcols[1]),
          [n2] "=&r" (newcols[2]), [n3] "=&r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]),
          [k0] "r" (round_key[0]), [k1] "r" (round_key[1]),
          [k2] "r" (round_key[2]), [k3] "r" (round_key[3]));
}

inline void __intrin_aes_last_dec_round(uint32_t *newcols,
                                        const uint32_t *oldcols,
                                        const uint32_t *round_key) {
    // aes32dsi
    asm volatile (
        "mv %[n0], %[k0]\n"
        "mv %[n1], %[k1]\n"
        "mv %[n2], %[k2]\n"
        "mv %[n3], %[k3]\n"
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
        : [n0] "+&r" (newcols[0]), [n1] "+&r" (newcols[1]),
          [n2] "+&r" (newcols[2]), [n3] "+&r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]),
          [k0] "r" (round_key[0]), [k1] "r" (round_key[1]),
          [k2] "r" (round_key[2]), [k3] "r" (round_key[3]));
}

// Hack to accelerate the InvMixColumns() invocations in the revised key
// schedule generation logic from Section 3.5.5 of the AES spec. We want
// to use the InvMixColumns() hardware implementation in aes32dsmi;
// however, it calls InvSubBytes() which we don't want. To work around
// this, we first perform SubBytes() via aes32esi and then
// InvSubBytes()+InvMixColumns() via aes32dsmi. Despite this being a
// grotesque kludge, it appears to beat our software InvMixColumns()
// implementation compiled with -O3, showing an ~8% reduction in total
// instruction count for our test kernel which decrypts 4KiB of data
// (split evenly across 16 threads)
inline void __intrin_aes_inv_mixcols(uint32_t *newcols, uint32_t *oldcols) {
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    asm volatile (
        // First, do subbytes on all the columns with aes32esi
        ".insn r 0x33, 0, 0x19, x0, %[s0], %[o0]\n"
        ".insn r 0x33, 0, 0x39, x0, %[s0], %[o0]\n"
        ".insn r 0x33, 0, 0x59, x0, %[s0], %[o0]\n"
        ".insn r 0x33, 0, 0x79, x0, %[s0], %[o0]\n"
        ".insn r 0x33, 0, 0x19, x0, %[s1], %[o1]\n"
        ".insn r 0x33, 0, 0x39, x0, %[s1], %[o1]\n"
        ".insn r 0x33, 0, 0x59, x0, %[s1], %[o1]\n"
        ".insn r 0x33, 0, 0x79, x0, %[s1], %[o1]\n"
        ".insn r 0x33, 0, 0x19, x0, %[s2], %[o2]\n"
        ".insn r 0x33, 0, 0x39, x0, %[s2], %[o2]\n"
        ".insn r 0x33, 0, 0x59, x0, %[s2], %[o2]\n"
        ".insn r 0x33, 0, 0x79, x0, %[s2], %[o2]\n"
        ".insn r 0x33, 0, 0x19, x0, %[s3], %[o3]\n"
        ".insn r 0x33, 0, 0x39, x0, %[s3], %[o3]\n"
        ".insn r 0x33, 0, 0x59, x0, %[s3], %[o3]\n"
        ".insn r 0x33, 0, 0x79, x0, %[s3], %[o3]\n"
        // Zero out destinations first, since the instructions below xor
        // into the result
        "andi %[n0], x0, 0\n"
        "andi %[n1], x0, 0\n"
        "andi %[n2], x0, 0\n"
        "andi %[n3], x0, 0\n"
        // Now do invsubbytes + invmixcols with aes32dsmi
        ".insn r 0x33, 0, 0x1f, x0, %[n0], %[s0]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n0], %[s0]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n0], %[s0]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n0], %[s0]\n"
        ".insn r 0x33, 0, 0x1f, x0, %[n1], %[s1]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n1], %[s1]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n1], %[s1]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n1], %[s1]\n"
        ".insn r 0x33, 0, 0x1f, x0, %[n2], %[s2]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n2], %[s2]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n2], %[s2]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n2], %[s2]\n"
        ".insn r 0x33, 0, 0x1f, x0, %[n3], %[s3]\n"
        ".insn r 0x33, 0, 0x3f, x0, %[n3], %[s3]\n"
        ".insn r 0x33, 0, 0x5f, x0, %[n3], %[s3]\n"
        ".insn r 0x33, 0, 0x7f, x0, %[n3], %[s3]"
        : [n0] "=&r" (newcols[0]), [n1] "=&r" (newcols[1]),
          [n2] "=&r" (newcols[2]), [n3] "=&r" (newcols[3]),
          [s0] "+&r" (s0), [s1] "+&r" (s1),
          [s2] "+&r" (s2), [s3] "+&r" (s3)
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]));
}

inline uint32_t __intrin_aes_subword(uint32_t word) {
    uint32_t ret = 0;
    asm volatile (
        // Use aes32esi for SubBytes
        ".insn r 0x33, 0, 0x19, x0, %[ret], %[word]\n"
        ".insn r 0x33, 0, 0x39, x0, %[ret], %[word]\n"
        ".insn r 0x33, 0, 0x59, x0, %[ret], %[word]\n"
        ".insn r 0x33, 0, 0x79, x0, %[ret], %[word]"
        : [ret] "+&r" (ret)
        : [word] "r" (word));

    return ret;
}

inline uint32_t __intrin_rotl(uint32_t word, uint32_t n) {
    uint32_t ret;
    asm volatile (
        ".insn r 0x33, 1, 0x30, %[ret], %[word], %[n]\n"
        : [ret] "=r" (ret)
        : [word] "r" (word), [n] "r" (n));

    return ret;
}

inline uint32_t __intrin_rotr(uint32_t word, uint32_t n) {
    uint32_t ret;
    asm volatile (
        ".insn r 0x33, 5, 0x30, %[ret], %[word], %[n]\n"
        : [ret] "=r" (ret)
        : [word] "r" (word), [n] "r" (n));

    return ret;
}

inline uint32_t __intrin_rotr_imm(uint32_t word, int32_t n) {
    uint32_t ret;
    asm volatile (
        ".insn i 0x13, 5, %[ret], %[word], %[n]\n"
        : [ret] "=r" (ret)
        : [word] "r" (word), [n] "i" ((0x30 << 5) | (n & 0x01f)));

    return ret;
}

#define __if(b) vx_split(b); \
                if (b) 

#define __else else

#define __endif vx_join();

#ifdef __cplusplus
}
#endif

#endif
