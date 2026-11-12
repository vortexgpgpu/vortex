#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// flash/flash_dxa use a single TYPE (default float) for both input and output
// but TCU has no fp32 multiplier so cannot have ITYPE fp32. OTYPE stays fp32
// since TCU accumulates in fp32 and softmax needs fp32.
#ifndef ITYPE
#define ITYPE tf32
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// Valid for WGMMA: 8, 16, 32
#ifndef WGMMA_NRC
#define WGMMA_NRC 8
#endif

typedef struct {
  uint32_t seq_len;
  uint32_t head_dim;
  uint32_t head_dim_tile;
  uint32_t block_size_r;
  uint32_t block_size_c;
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t V_addr;
  uint64_t O_addr;
} kernel_arg_t;

#endif
