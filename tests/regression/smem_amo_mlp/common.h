#ifndef SMEM_AMO_MLP_COMMON_H
#define SMEM_AMO_MLP_COMMON_H

#include <stdint.h>

enum smem_amo_mode_t : uint32_t {
  SMEM_AMO_SAME_BANK = 0,
  SMEM_AMO_ALL_BANKS = 1,
  SMEM_PRIVATE_RMW   = 2,
  SMEM_AMO_DIRECTED  = 3,
};

struct kernel_arg_t {
  uint32_t mode;
  uint32_t iters;
  uint32_t num_harts;
  uint32_t num_banks;
  uint64_t old_sums_addr;
  uint64_t final_values_addr;
};

#endif
