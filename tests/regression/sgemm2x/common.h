#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE int
#endif

typedef struct {
  uint32_t num_groups;
  uint32_t group_size;
  uint32_t size;
  uint32_t tile_size;
  uint64_t local_addr;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
