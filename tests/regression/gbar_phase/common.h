#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Global-barrier ids used by the kernel.
//
// vortex::gbarrier(id) packs rs1 = (id << 8) | 0x80000000, so the hardware
// derives BOTH the cluster's gbar row (rs1[8 +: NB_BITS]) and the per-core
// barrier slot ({rs1[NW_BITS-1:0], rs1[8 +: NB_BITS]}) from `id`. Two ids are
// used so the rendezvous never shares state with the barrier under probe.
#define GBAR_PROBE_ID 1   // rs1 = 0x80000100 -> gbar row 1, per-core slot 1
#define GBAR_SYNC_ID  2   // rs1 = 0x80000200 -> gbar row 2, per-core slot 2

typedef struct {
  uint32_t num_cores;
  uint32_t num_groups;
  uint32_t group_size;
  uint64_t p0_addr;   // uint32_t per core: probe phase BEFORE a generation
  uint64_t p1_addr;   // uint32_t per core: probe phase AFTER that generation
} kernel_arg_t;

#endif // _COMMON_H_
