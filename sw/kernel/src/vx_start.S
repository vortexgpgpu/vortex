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
#include <newlib.h>
#include "common.h"

#ifdef VM_ENABLE
// Mode encodings used by the SATP CSR. Defined locally to avoid pulling
// sim/common/mem.h into the kernel build (these enum values are also
// declared there).
#define BARE 0
#define SV32 1
#define SV39 8
#endif

.section .init, "ax"
.global _start
.type   _start, @function
_start:

#ifdef KMU_ENABLE

#ifdef VM_ENABLE
  # Program SATP with the page-table base PPN and addressing mode. The
  # runtime has already populated the page table at PAGE_TABLE_BASE_ADDR
  # before launching this kernel; we just need to point the per-core MMU
  # at it. STACK and PT regions are MMU-bypass per VMManager::need_trans,
  # so it is safe to do this before the stack is set up. The non-KMU
  # path writes SATP from init_regs (every spawned warp).
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

#ifdef NEED_GP
  # set global pointer register
  la    gp, __global_pointer
#endif

  # set stack pointer register
  LOAD_IMMEDIATE64(sp, STACK_BASE_ADDR)
  csrr  t0, VX_CSR_MHARTID
  sll   t1, t0, STACK_LOG2_SIZE
  sub   sp, sp, t1

#ifdef NEED_TLS
  # set thread pointer register
  # use address space after BSS region
  la    t1, __tbss_size
  mul   t0, t0, t1
  la    tp, _end
  add   tp, tp, t0

  # initialize TLS for all warps
  call  __init_tls
#endif

#ifdef NEED_INITFINI
  # run global initialization functions
  call  __libc_init_array
#endif

  # call main program routine
  csrr  a0, VX_CSR_MSCRATCH
  call  kernel_main

  # shutdown warp
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0

#else

  # initialize per-thread registers
  csrr  t0, VX_CSR_NUM_WARPS  # get num warps
  la    t1, init_regs_all
  .insn r RISCV_CUSTOM0, 1, 0, x0, t0, t1  # wspawn t0, t1
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  jal   init_regs
  li    t0, 1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0

#ifdef NEED_TLS
  # initialize TLS for all warps
  csrr  t0, VX_CSR_NUM_WARPS  # get num warps
  la    t1, init_tls_all
  .insn r RISCV_CUSTOM0, 1, 0, x0, t0, t1  # wspawn t0, t1
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  call  __init_tls
  li    t0, 1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
#endif


#ifdef NEED_INITFINI
  # run global initialization functions
  call  __libc_init_array
#endif

  # call main program routine
  call  main

  # call exit routine
  tail  exit
.size _start, .-_start

.section .text
.type _Exit, @function
.global _Exit
_Exit:
  li t0, IO_EXIT_CODE
  sw a0, 0(t0)
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0

.section .text
.type init_regs, @function
.local init_regs
init_regs:
#ifdef VM_ENABLE
  # SATP is per-core, but each warp on each core writes the same value
  # idempotently — keeps the non-KMU init path symmetric with _start so
  # spawned warps don't rely on warp-0's earlier write being visible
  # before any user-VA access they make.
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

#ifdef NEED_GP
  # set global pointer register
  .option push
  .option norelax
  la    gp, __global_pointer
  .option pop
#endif

  # set stack pointer register
  LOAD_IMMEDIATE64(sp, STACK_BASE_ADDR)
  csrr  t0, VX_CSR_MHARTID
  sll   t1, t0, STACK_LOG2_SIZE
  sub   sp, sp, t1

#ifdef NEED_TLS
  # set thread pointer register
  # use address space after BSS region
  la    t1, __tbss_size
  mul   t0, t0, t1
  la    tp, _end
  add   tp, tp, t0
#endif
  ret

.section .text
.type init_regs_all, @function
.local init_regs_all
init_regs_all:
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  jal   init_regs
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0
  ret

#ifdef NEED_TLS
.section .text
.type init_tls_all, @function
.local init_tls_all
init_tls_all:
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  call  __init_tls
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0
  ret
#endif

#endif

.section .data
	
	.weak __dso_handle
__dso_handle:
	.long	0
