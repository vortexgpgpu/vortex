#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>

typedef struct {
  uint32_t count;
  uint32_t round;
  uint32_t rotate;
  uint32_t burst;
  uint32_t dxa_slot;   // 1 + DXA descriptor slot of src, 0 to load with the LSU
  uint64_t src_addr;
  uint64_t dst_addr;
  uint64_t acc_addr;
  uint64_t tail_addr;
} kernel_arg_t;

inline uint32_t tail_value(uint32_t i, uint32_t k, uint32_t round) {
  return (i * 0x01000193u) ^ (k << 20) ^ (round * 0x2545F491u);
}

#endif
