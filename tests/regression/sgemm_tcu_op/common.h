#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#ifdef VX_CFG_NUM_THREADS
#define NUM_THREADS VX_CFG_NUM_THREADS
#else
#define NUM_THREADS 4
#endif
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// C (the accumulator input of D = A*B + C) exists only when the test fills it
// with real data (-DSGEMM_NONZERO_C), or when -DSGEMM_FORCE_C_BUFFER keeps a
// zero-filled C for A/B comparisons. Otherwise there is no C (beta == 0): the
// host passes a null C_addr and the kernel neither stages C nor hands the
// engine one. A compile-time constant shared by host and kernel, so both agree
// and the C-present kernel is the same code it always was: the kernel is fully
// unrolled, and even a few extra instructions per tile shift its icache-line
// layout enough to move cycle counts by several percent.
#if defined(SGEMM_NONZERO_C) || defined(SGEMM_FORCE_C_BUFFER)
#define SGEMM_HAS_C 1
#else
#define SGEMM_HAS_C 0
#endif

// Multi-engine TCU_OP (dense only): the kernel runs as SGEMM_TCU_ENGINES
// single-warp CTAs. CTA c owns output tiles c, c+E, c+2E, ... and runs on warp
// c, which maps to issue slot c % ISSUE_WIDTH and hence to TCU engine c. Each
// CTA gets its own LMEM share (two double-buffered stages of A + B [+ C]) and,
// because barrier ids carry the CTA's local group id, its own barriers.
// 1 is today's single-engine kernel, unchanged.
#ifndef SGEMM_TCU_ENGINES
#define SGEMM_TCU_ENGINES 1
#endif

// Cooperative multi-engine (-DSGEMM_TCU_COOP, needs SGEMM_TCU_ENGINES 2 or 4):
// ONE CTA of SGEMM_TCU_ENGINES warps instead of one CTA per engine. Warp w
// still lands on issue slot w and drives engine w, but the warps share one
// LMEM allocation and one barrier set and cover a GM x GN block of output
// tiles (1x2 for 2 engines, 2x2 for 4). Each stage stages GM A tiles + GN B
// tiles that the engines share, instead of one A + one B per engine: half the
// DRAM traffic at 4 engines, and double-buffered tile_K=128 fits in 64KB.
#ifdef SGEMM_TCU_COOP
#if (SGEMM_TCU_ENGINES == 2)
#define SGEMM_COOP_GM 1
#define SGEMM_COOP_GN 2
#elif (SGEMM_TCU_ENGINES == 4)
#define SGEMM_COOP_GM 2
#define SGEMM_COOP_GN 2
#else
#error "SGEMM_TCU_COOP needs SGEMM_TCU_ENGINES of 2 or 4"
#endif
#define SGEMM_COOP_WARPS SGEMM_TCU_ENGINES
#else
#define SGEMM_COOP_WARPS 1
#endif

// Per-CTA LMEM bytes for the multi-engine kernel; host and kernel must agree
// (the host passes it as the launch's lmem_size, the kernel as __local_mem).
#define SGEMM_CTA_LMEM_BYTES(elem_bytes, tile_k, has_c) \
  (2u * (2u * 32u * (tile_k) * (elem_bytes) + ((has_c) ? 32u * 32u * 4u : 0u)))

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t D_addr;
  uint64_t A_bitmap_addr;
  uint64_t B_bitmap_addr;
  uint32_t max_a_blocks;
  uint32_t max_b_blocks;
  uint8_t  sparsity;            
  uint64_t metrics_addr;
} kernel_arg_t;

#endif
