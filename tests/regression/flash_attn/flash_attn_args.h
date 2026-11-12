// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// Kernel arguments shared by the flash_attn tests.

#ifndef FLASH_ATTN_ARGS_H
#define FLASH_ATTN_ARGS_H

#include <stdint.h>

typedef struct {
  uint32_t seq_len;
  uint32_t head_dim;
  uint32_t head_dim_tile;
  uint32_t block_size_r;
  uint32_t block_size_c;
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t V_addr;
  uint64_t O_addr;
} kernel_arg_t;

#endif // FLASH_ATTN_ARGS_H
