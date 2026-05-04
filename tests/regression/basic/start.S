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

#include <VX_config.h>
#include <VX_types.h>

#ifdef VM_ENABLE
// Mode encodings used by SATP. Defined locally to avoid pulling
// sim/common/mem.h into the kernel build.
#define BARE 0
#define SV32 1
#define SV39 8
#endif

.section .init, "ax"
.global _start
.type _start, @function
_start:
#ifdef VM_ENABLE
  # Each core programs SATP from the runtime-installed page table at
  # PAGE_TABLE_BASE_ADDR. Must run before any user-VA load/store.
#if VM_ADDR_MODE == SV39
  li    t0, PAGE_TABLE_BASE_ADDR
  srli  t0, t0, MEM_PAGE_LOG2_SIZE
  li    t1, 1
  slli  t1, t1, 63
  or    t0, t0, t1
  csrw  satp, t0
#elif VM_ADDR_MODE == SV32
  li    t0, PAGE_TABLE_BASE_ADDR
  srli  t0, t0, MEM_PAGE_LOG2_SIZE
  li    t1, 1
  slli  t1, t1, 31
  or    t0, t0, t1
  csrw  satp, t0
#endif
#endif

  # call main routine
  call main

  # end execution
  .insn r 0x0b, 0, 0, x0, x0, x0
.size _start, .-_start