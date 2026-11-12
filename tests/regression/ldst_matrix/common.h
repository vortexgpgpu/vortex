#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Each thread owns a source and a destination region in global memory and in
// local memory; a fragment's row groups sit inside them.
#define GLOBAL_REGION_WORDS 512
#define LOCAL_REGION_WORDS  128

// Source word i of a thread's region; local and global words differ.
static inline uint32_t src_word(uint32_t tid, uint32_t i, uint32_t salt) {
  return ((tid << 12) | i | 0x40000000u) ^ salt;
}

#define GLOBAL_SALT 0x00800000u
#define LOCAL_SALT  0x01000000u

typedef struct {
  uint64_t src_addr;   // global source regions, filled by the host
  uint64_t dst_addr;   // global destination regions, zeroed by the host
  uint64_t err_addr;   // out: per-thread error count
  uint64_t shape_addr; // out: bit s set when shape s ran
} kernel_arg_t;

#endif
