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

#define RISCV_CUSTOM0   0x0B

.section .init, "ax"
.global _start
.type   _start, @function
_start:  

  # initialize per-thread registers
  csrr  t0, VX_CSR_NUM_WARPS  # get num warps
  la    t1, init_regs_all
  .insn r RISCV_CUSTOM0, 1, 0, x0, t0, t1  # wspawn t0, t1
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  jal   init_regs
  li    t0, 1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0

  # wait for spawn warps to terminate
  jal vx_wspawn_wait

  # initialize TLS for all warps
  csrr  t0, VX_CSR_NUM_WARPS  # get num warps
  la    t1, init_tls_all
  .insn r RISCV_CUSTOM0, 1, 0, x0, t0, t1  # wspawn t0, t1
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  call  __init_tls
  li    t0, 1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0

  # wait for spawn warps to terminate
  jal vx_wspawn_wait
  
  # clear BSS segment
  la    a0, _edata
  la    a2, _end
  sub   a2, a2, a0
  li    a1, 0
  call  memset

  # initialize trap vector
  # la t0, trap_entry
  # csrw mtvec, t0

  # register global termination functions
  la    a0, __libc_fini_array   
  call  atexit                  

  # run global initialization functions
  call  __libc_init_array  

  # call main program routine
  call  main

  # call exit routine
  tail  exit
.size _start, .-_start

.section .text
.type _exit, @function
.global _exit
_exit:
  mv    s0, a0
  call  vx_perf_dump 
  mv    gp, s0
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0

.section .text
.type init_regs, @function
.local init_regs
init_regs:
  # set global pointer register
  .option push
  .option norelax
  la    gp, __global_pointer
  .option pop
  
  # set stack pointer register
#if (XLEN == 64)
  li    t0, (STACK_BASE_ADDR >> 32)
  slli  t0, t0, 32
  li    sp, (STACK_BASE_ADDR & 0xffffffff)
  or    sp, sp, t0
#else
  li    sp, STACK_BASE_ADDR # load stack base address
#endif
  csrr  t0, VX_CSR_MHARTID
  sll   t1, t0, STACK_LOG2_SIZE
  sub   sp, sp, t1

  # set thread pointer register
  # use address space after BSS region
  # ensure cache line alignment
  la    t1, __tcb_aligned_size
  mul   t0, t0, t1
  la    tp, _end + 63
  add   tp, tp, t0
  and   tp, tp, -64
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

.section .text
.type init_tls_all, @function
.local init_tls_all
init_tls_all:
  li    t0, -1
  .insn r RISCV_CUSTOM0, 0, 0, x0, t0, x0  # tmc t0
  call  __init_tls
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0
  ret 

.section .text
.type vx_wspawn_wait, @function
.global vx_wspawn_wait
vx_wspawn_wait:
  csrr  t0, VX_CSR_WARP_MASK
  li    t1, 1
  bne   t0, t1, vx_wspawn_wait
  ret

.section .data
	.global __dso_handle
	.weak __dso_handle
__dso_handle:
	.long	0
