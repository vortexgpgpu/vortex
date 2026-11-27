#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// K tiles per iteration of the K loop
#ifndef KUNROLL
#define KUNROLL 1
#endif

// Row-major A (WGMMA_RMAJOR_A) and K-major B (WGMMA_KMAJOR_B): rows padded by
// *_PAD_K K-slices, each operand starting *_OFF_K K-slices into its row.
#ifndef A_PAD_K
#define A_PAD_K 0
#endif
#ifndef A_OFF_K
#define A_OFF_K 0
#endif
// Extra A offset in elements: one that is not a multiple of the K-slice builds
// a descriptor the TCU cannot fetch, which SimX must reject.
#ifndef A_OFF_ELEMS
#define A_OFF_ELEMS 0
#endif
#ifndef B_PAD_K
#define B_PAD_K 0
#endif
#ifndef B_OFF_K
#define B_OFF_K 0
#endif
// Extra per-CTA LMEM in bytes: a footprint that is not a multiple of an LMEM
// bank row still gives every co-resident CTA a bank-row aligned base.
#ifndef SMEM_PAD_BYTES
#define SMEM_PAD_BYTES 0
#endif
// B_RELAYOUT: even K tiles stage B K-major, odd K tiles block-major at a base
// one LMEM bank row short of a B tile further on, so the first bank row an odd
// tile reads is the one the previous WGMMA's last B fetch was keyed on.
#ifndef B_RELAYOUT
#define B_RELAYOUT 0
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
