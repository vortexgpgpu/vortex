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

//
// AES-256
//
inline void aes_enc_round(uint32_t *newcols, uint32_t *oldcols) {
    // aes32esmi
    asm volatile (
        // See:
        // https://sourceware.org/binutils/docs-2.36/as/RISC_002dV_002dFormats.html
        // The last operation here is weird because it's a signed
        // immediate. If you print it as binary and compare with the
        // draft spec, it will make more sense
        ".insn s 0x33, 0, %[n0],   864(%[o0])\n"
        ".insn s 0x33, 0, %[n0],  1888(%[o1])\n"
        ".insn s 0x33, 0, %[n0], -1184(%[o2])\n"
        ".insn s 0x33, 0, %[n0],  -160(%[o3])\n"
        ".insn s 0x33, 0, %[n1],   864(%[o1])\n"
        ".insn s 0x33, 0, %[n1],  1888(%[o2])\n"
        ".insn s 0x33, 0, %[n1], -1184(%[o3])\n"
        ".insn s 0x33, 0, %[n1],  -160(%[o0])\n"
        ".insn s 0x33, 0, %[n2],   864(%[o2])\n"
        ".insn s 0x33, 0, %[n2],  1888(%[o3])\n"
        ".insn s 0x33, 0, %[n2], -1184(%[o0])\n"
        ".insn s 0x33, 0, %[n2],  -160(%[o1])\n"
        ".insn s 0x33, 0, %[n3],   864(%[o3])\n"
        ".insn s 0x33, 0, %[n3],  1888(%[o0])\n"
        ".insn s 0x33, 0, %[n3], -1184(%[o1])\n"
        ".insn s 0x33, 0, %[n3],  -160(%[o2])"
        : [n0] "+r" (newcols[0]), [n1] "+r" (newcols[1]),
          [n2] "+r" (newcols[2]), [n3] "+r" (newcols[3])
        : [o0] "r" (oldcols[0]), [o1] "r" (oldcols[1]),
          [o2] "r" (oldcols[2]), [o3] "r" (oldcols[3]));
}

inline void aes_last_enc_round(uint32_t *newcols, uint32_t *oldcols) {
    // aes32esi
    asm volatile (
        // See:
        // https://sourceware.org/binutils/docs-2.36/as/RISC_002dV_002dFormats.html
        // The last operation here is weird because it's a signed
        // immediate. If you print it as binary and compare with the
        // draft spec, it will make more sense
        ".insn s 0x33, 0, %[n0],   800(%[o0])\n"
        ".insn s 0x33, 0, %[n0],  1824(%[o1])\n"
        ".insn s 0x33, 0, %[n0], -1248(%[o2])\n"
        ".insn s 0x33, 0, %[n0],  -224(%[o3])\n"
        ".insn s 0x33, 0, %[n1],   800(%[o1])\n"
        ".insn s 0x33, 0, %[n1],  1824(%[o2])\n"
        ".insn s 0x33, 0, %[n1], -1248(%[o3])\n"
        ".insn s 0x33, 0, %[n1],  -224(%[o0])\n"
        ".insn s 0x33, 0, %[n2],   800(%[o2])\n"
        ".insn s 0x33, 0, %[n2],  1824(%[o3])\n"
        ".insn s 0x33, 0, %[n2], -1248(%[o0])\n"
        ".insn s 0x33, 0, %[n2],  -224(%[o1])\n"
        ".insn s 0x33, 0, %[n3],   800(%[o3])\n"
        ".insn s 0x33, 0, %[n3],  1824(%[o0])\n"
        ".insn s 0x33, 0, %[n3], -1248(%[o1])\n"
        ".insn s 0x33, 0, %[n3],  -224(%[o2])"
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
