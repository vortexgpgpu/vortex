#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
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
  uint64_t D_addr;              
  uint64_t A_bitmap_addr;       
  uint64_t B_bitmap_addr;       
  // uint64_t A_nz_addr;           
  // uint64_t B_nz_addr;           
  // uint32_t A_compressed_blocks; 
  // uint32_t B_compressed_blocks; 
  uint32_t max_a_blocks;
  uint32_t max_b_blocks;
  uint8_t  sparsity;            
  uint64_t metrics_addr;
} kernel_arg_t;

#endif
