#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define UNIFORM_MARK 0x80000000u

// Per warp and iteration: one RAW word and one WAR word (four words apart,
// so four iterations share a line), and one 16-word line of read-only words.
#define RAW_STRIDE 4
#define WAR_STRIDE 4
#define RO_LINE    16

typedef struct {
  uint32_t num_iters;   // iterations per warp
  uint32_t num_threads; // total threads launched
  uint32_t warp_size;   // threads per warp
  uint32_t scatter_words;
  uint64_t raw_addr;    // in: initial words; out: RAW words overwritten with UNIFORM_MARK + index
  uint64_t war_addr;    // in: initial words; out: WAR words overwritten with UNIFORM_MARK + index
  uint64_t ro_addr;     // in: read-only words
  uint64_t scatter_addr;// in: read-only words scattered across lines
  uint64_t err_addr;    // out: per-thread error count
  uint64_t sink_addr;   // out: per-thread sum of the scattered loads (keeps them alive)
} kernel_arg_t;

#endif
