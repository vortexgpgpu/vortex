#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

//==============================================================================
// DFV (Design-for-Verification) CSR Definitions
//==============================================================================
// These CSRs allow software control of hardware verification hooks
#define VX_CSR_DFV_CTRL           0x7C0  // DFV global enable (bit 0)
#define VX_CSR_DFV_ICACHE_STALL   0x7C1  // ICache LFSR-based stall enable (bit 0)
#define VX_CSR_DFV_RANDOM_SEED    0x7C2  // LFSR seed (32-bit) for reproducible randomness
#define VX_CSR_DFV_SET_THRESHOLD   0x7C3 // Stall probability (0-255): stall when lfsr[7:0] < threshold
                                          // Examples: 0=never stall, 128=~50%, 255=always stall
#define VX_CSR_DFV_DCACHE_STALL   0x7C4  // DCache request LFSR-based stall enable (bit 0)
#define VX_CSR_DFV_WRITEBACK_STALL 0x7C5 // Writeback stall enable (bit 0)
#define VX_CSR_DFV_FILL_STALL      0x7C6 // Cache fill stall enable (bit 0)
#define VX_CSR_DFV_RELEASE_THRESHOLD 0x7C7 // Release probability (0-255)
#define VX_CSR_DFV_RELEASE_SEED    0x7C8 // LFSR2 seed for release timing
#define VX_CSR_DFV_RELEASE_DELAY   0x7C9 // Per-point release delay [3:0]=ic [7:4]=dc [11:8]=wb [15:12]=fill
#define VX_CSR_DFV_RELEASE_FOREVER 0x7CA // When 1: once released, stalls stay off permanently
#define VX_CSR_DFV_THROTTLE_THRESHOLD 0x7CB // Throttle counter threshold (16-bit)

typedef struct {
  uint32_t grid_dim[2];
  uint32_t width;
  uint64_t I_addr;
  uint64_t W_addr;
  uint64_t O_addr;
  uint32_t use_lmem;
  uint32_t enable_dfv_test;  // If non-zero, enable DFV stress testing
} kernel_arg_t;

#endif
