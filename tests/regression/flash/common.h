#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
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
