// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <stdint.h>
typedef struct {
  uint32_t rounds;
  uint32_t shared;     // 1: the exchange buffer is workgroup memory, 0: global
  uint32_t junk;       // stores to other lines each thread makes first
  uint32_t mode;       // 1: only the even warps store ahead, to the exchange line's bank
  uint64_t junk_addr;  // uint32[gridDim.x * blockDim.x * 16 * junk]
  uint64_t buf_addr;   // uint32[2 * blockDim.x] per CTA, then the mode-1 stores
  uint64_t err_addr;   // uint32[gridDim.x * blockDim.x], stale reads per thread
} kernel_arg_t;
