#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// One result per probe: the trap cause (-1 if none) and the faulting pc
// relative to the probed instruction.
enum {
  PROBE_FS_OFF_FADD = 0,  // FP instructions with mstatus.FS=Off
  PROBE_FS_OFF_FLW,
  PROBE_FS_OFF_FFLAGS,
  PROBE_FS_OFF_FMV,
  PROBE_BRANCH_FUNCT3_2,  // reserved encodings
  PROBE_BRANCH_FUNCT3_3,
  PROBE_SYSTEM_RESERVED,
  PROBE_SYSTEM_FUNCT3_4,
  PROBE_WFI,              // runs as a no-op
  NUM_PROBES,
};

typedef struct {
  int64_t cause;
  int64_t epc_offset;
} probe_t;

typedef struct {
  probe_t probes[NUM_PROBES];
  uint64_t mstatus_initial; // FS set to Initial, before any FP write
  uint64_t mstatus_dirty;   // after an FP write
} trap_out_t;

typedef struct {
  uint64_t dst_addr;
} kernel_arg_t;

#endif
