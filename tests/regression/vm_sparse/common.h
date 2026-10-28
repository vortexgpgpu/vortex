#ifndef _COMMON_H_
#define _COMMON_H_

// Every probed page holds SLOTS slots, SLOT_STRIDE words apart, so a page's
// slots sit on different cache lines. A slot is three words: one read, one
// stored, one atomically incremented.
#define SPARSE_PAGES  16u
#define SLOTS         16u
#define SLOT_STRIDE   64u

typedef struct {
  uint64_t sparse_addr;   // the reservation
  uint64_t read_addr;     // per-thread word read from its slot
  uint64_t amo_addr;      // per-thread value its atomic add returned
  uint32_t tag;           // stored value = tag | thread
  uint32_t num_threads;   // SPARSE_PAGES * SLOTS
  uint64_t count_addr;    // nonzero: each launched thread only counts itself here
} kernel_arg_t;

#endif
