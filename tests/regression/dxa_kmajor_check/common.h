// DXA writer + multi-descriptor isolation test.
// Each CTA programs two 2D descriptors:
//   A descriptor: row-major (LAYOUT=ROW_MAJOR) — A[m][k] → A_smem[m*tileK + k]
//   B descriptor: K-major  (LAYOUT=K_MAJOR)   — B[k][n] → B_smem[n*tileK + k]
// Kernel byte-copies both SMEM regions to dst; host byte-diffs.

#pragma once
#include <stdint.h>

using elem_t = uint16_t;

struct kernel_arg_t {
  uint64_t srcA_addr;
  uint64_t srcB_addr;
  uint64_t dst_addr;
  uint32_t N;
  uint32_t K;
  uint32_t M;
  uint32_t tileN;
  uint32_t tileK;
  uint32_t tileM;       // M rows in A tile (per CTA)
  uint32_t a_bytes;     // tileM * tileK * sizeof(elem_t)
  uint32_t b_bytes;     // tileN * tileK * sizeof(elem_t)
  uint32_t cta_bytes;   // a_bytes + b_bytes (per-CTA dst slot)
};
