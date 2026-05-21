#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_config.h>

#ifdef TCU_WG_N_MUL
  #ifdef WGMMA_NRC
    #if WGMMA_NRC != 8 * TCU_WG_N_MUL
      #error "WGMMA_NRC conflicts with TCU_WG_N_MUL; do not set WGMMA_NRC when TCU_WG_N_MUL is defined"
    #endif
  #else
    #define WGMMA_NRC (8 * TCU_WG_N_MUL)
  #endif
#else
  #ifndef WGMMA_NRC
    #define WGMMA_NRC 8
  #endif
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
