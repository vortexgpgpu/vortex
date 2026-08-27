#ifndef _COMMON_H_
#define _COMMON_H_

// Slots in the compare-and-swap ladder. The shared buffer is sized from the
// hart count, and one hart still gives 64 bytes, so 16 words always fit
// whatever the launch shape or the iteration count.
#define CAS_LADDER_SLOTS 16

// Args shared between host (main.cpp) and the GPU kernel (kernel.cpp).
// Layout matches dogfood's, with `iters` added for the contention loops.
typedef struct {
  uint32_t testid;
  uint32_t num_harts;     // total harts hammering the shared word
  uint32_t iters;         // per-hart iteration count
  uint64_t shared_addr;   // single-word AMO target (4 or 8 bytes)
  uint64_t per_hart_addr; // per-hart scratch / observed-old buffer
} kernel_arg_t;

#endif
