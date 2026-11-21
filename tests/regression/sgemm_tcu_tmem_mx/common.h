#ifndef _SGEMM_TCU_TMEM_MX_COMMON_H_
#define _SGEMM_TCU_TMEM_MX_COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
#define WGMMA_NRC 8
#endif
#ifndef ITYPE
#define ITYPE mxfp8
#endif
#ifndef OTYPE
#define OTYPE fp32
#endif

// Shared memory starts with the CTA's TMEM column, padded to keep the tiles aligned.
#define TMEM_SLOT_BYTES 64

typedef struct {
  uint32_t M, N, K;          // K in storage bytes
  uint32_t sf_cols;          // TMEM columns of one K step's scales
  uint32_t img_words;        // words of one CTA's scale image
  uint64_t A_addr, B_addr, C_addr;
  uint64_t SF_addr;          // the CTAs' scale images, tc.cp storage order
#ifdef TCU_MX_TLS
  uint64_t A_tensor_scale_addr, B_tensor_scale_addr;
#endif
} kernel_arg_t;

#endif
