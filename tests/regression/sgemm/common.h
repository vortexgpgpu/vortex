#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

// LDST_WIDE=1 reads each thread's consecutive words with the wide loads
// (vx_flw_q); 0 keeps scalar accesses.
#ifndef LDST_WIDE
#define LDST_WIDE 0
#endif

typedef struct {
  uint32_t size;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
