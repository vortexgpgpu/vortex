#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Barrier ids. vortex::group_barrier(id) packs rs1 = (id << 8), and the
// hardware slot address is {rs1[NW_BITS-1:0], rs1[8 +: NB_BITS]}, so these
// three ids are three DISTINCT per-core barrier slots.
#define GATE_ID   2   // rendezvous: releases every warp of the core at once
#define PROBE0_ID 0   // warp 0 probes this slot
#define PROBE1_ID 1   // warp 1 probes this slot

#define ROUNDS 256

typedef struct {
  uint32_t num_cores;
  uint32_t num_warps;
  uint32_t rounds;
  uint64_t err_addr;   // uint32_t per (core,warp): rounds where the phase failed to flip
  uint64_t pre_addr;   // uint32_t per (core,warp): first observed `pre`  on failure
  uint64_t post_addr;  // uint32_t per (core,warp): first observed `post` on failure
} kernel_arg_t;

#endif // _COMMON_H_
