#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE int
#endif

//==============================================================================
// DFV (Design-for-Verification) CSR Definitions
//==============================================================================
#define VX_CSR_DFV_CTRL           0x7C0
#define VX_CSR_DFV_ICACHE_STALL   0x7C1
#define VX_CSR_DFV_RANDOM_SEED    0x7C2
#define VX_CSR_DFV_SET_THRESHOLD   0x7C3
#define VX_CSR_DFV_DCACHE_STALL   0x7C4
#define VX_CSR_DFV_WRITEBACK_STALL 0x7C5
#define VX_CSR_DFV_FILL_STALL      0x7C6
#define VX_CSR_DFV_RELEASE_THRESHOLD 0x7C7
#define VX_CSR_DFV_RELEASE_SEED    0x7C8
#define VX_CSR_DFV_RELEASE_DELAY   0x7C9
#define VX_CSR_DFV_RELEASE_FOREVER 0x7CA // When 1: once released, stalls stay off permanently
#define VX_CSR_DFV_THROTTLE_THRESHOLD 0x7CB // Throttle counter threshold (16-bit)

//==============================================================================
// DFV Stress Test Configuration
//==============================================================================
// The test runs multiple phases with different DFV configs to exercise
// various contention scenarios. Each phase does the same computation
// (so output verification catches any DFV-induced corruption), but with
// different stall combinations and thresholds.

#define DFV_NUM_PHASES 4

typedef struct {
  uint32_t num_points;
  uint32_t stride;         // Stride between accesses (controls cache conflict rate)
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;
  uint32_t enable_dfv_test;
  uint32_t dfv_phase;      // Which DFV config phase to run (0-3)
} kernel_arg_t;

#endif
