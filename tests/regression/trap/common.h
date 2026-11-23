#ifndef _COMMON_H_
#define _COMMON_H_

// Per-thread result bits; zero means every check passed.
#define TRAP_BAD_CAUSE 0x1u // mcause is not this warp's own exception code
#define TRAP_BAD_EPC   0x2u // mepc did not return to this warp's own trap site
#define TRAP_BAD_TVEC  0x4u // mtvec does not read back the block's handler address
#define TRAP_BAD_TVAL  0x8u // mtval is not zero

typedef struct {
  uint32_t num_points;
  uint32_t iters;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
