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

#ifdef VX_CFG_VM_ENABLE
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

// ===========================================================================
// KMU entry model.
//
// The KMU launches every (block, thread) coordinate of a kernel at that
// kernel's entry PC. There is exactly one real entry routine —
// __vx_cta_entry — which brings the warp up to a C-callable state
// (SATP / gp / sp / tp+TLS / global ctors), then dispatches to the kernel
// whose address the caller left in s11. s11 is callee-saved, so it
// survives the prologue's __init_tls / __libc_init_array calls. The
// kernel-arguments pointer is handed to the kernel in a0 (MSCRATCH).
//
// Two ways the KMU reaches __vx_cta_entry:
//
//   * Multi-entry .vxbin — each kernel has its own entry stub, emitted
//     into the .vx_entry section and listed in the VXSYMTAB footer:
//             lla s11, <kernel> ; j __vx_cta_entry
//     The KMU enters those stubs directly, one PC per named kernel.
//
//   * Footer-less .vxbin — the KMU enters at min_vma, where _start sits.
//     _start is itself just the entry stub for the kernel conventionally
//     named `kernel_main` (the legacy single-kernel-per-program model).
//
// So _start and every per-kernel stub are the same two-instruction shape
// over one shared prologue. kernel_main is weak: a footer-less program
// (e.g. the regression tests) defines it; a multi-entry program does not,
// and its dead _start simply resolves s11 to 0.
// ===========================================================================

  # _start — the legacy single-entry stub: route the footer-less
  # `kernel_main` convention through the shared CTA entry below.
  .weak kernel_main
  lla   s11, kernel_main
  j     __vx_cta_entry
  .size _start, .-_start

  # __vx_cta_entry — the one shared per-CTA entry routine.
  .global __vx_cta_entry
  .type   __vx_cta_entry, @function
__vx_cta_entry:
#ifdef VX_CFG_VM_ENABLE
  # Program SATP with the page-table base PPN and addressing mode. The
  # runtime has already populated the page table at VX_CFG_PAGE_TABLE_BASE_ADDR
  # before launching this kernel; we just point the per-core MMU at it.
  # STACK and PT regions are MMU-bypass per VMManager::need_trans, so it
  # is safe to do this before the stack is set up.
#if VX_CFG_VM_ADDR_MODE == SV39
  li    t0, VX_CFG_PAGE_TABLE_BASE_ADDR
  srli  t0, t0, VX_CFG_MEM_PAGE_LOG2_SIZE
  li    t1, 1
  slli  t1, t1, 63
  or    t0, t0, t1
  csrw  satp, t0
#elif VX_CFG_VM_ADDR_MODE == SV32
  li    t0, VX_CFG_PAGE_TABLE_BASE_ADDR
  srli  t0, t0, VX_CFG_MEM_PAGE_LOG2_SIZE
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
  LOAD_IMMEDIATE64(sp, VX_CFG_STACK_BASE_ADDR)
  csrr  t0, VX_CSR_MHARTID
  sll   t1, t0, VX_CFG_STACK_LOG2_SIZE
  sub   sp, sp, t1
#ifdef NEED_TLS
  # set thread pointer register: each hart gets its own TLS image in the
  # address space after the BSS region. The image spans .tdata + .tbss,
  # so the per-hart stride is __tls_block_size (= __tbss_offset +
  # __tbss_size), not __tbss_size alone — otherwise adjacent harts' TLS
  # images overlap by __tbss_offset bytes. __tls_block_size is a small
  # absolute linker symbol; address it with lui+addi rather than `la`,
  # whose medany PC-relative expansion overflows on RV64.
  lui   t1, %hi(__tls_block_size)
  addi  t1, t1, %lo(__tls_block_size)
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
  # dispatch to the kernel: entry address in s11, kargs pointer in a0.
  csrr  a0, VX_CSR_MSCRATCH
  # temporarily disable RVC to force a 4-byte call — the scheduler's CTA
  # rewind logic needs the dispatch site to be exactly 4 bytes.
  .option push
  .option norvc
  jalr  ra, s11
  .option pop
  # shutdown warp.
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0
  .size __vx_cta_entry, .-__vx_cta_entry

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
  li t0, VX_CFG_IO_EXIT_CODE
  sw a0, 0(t0)
  .insn r RISCV_CUSTOM0, 0, 0, x0, x0, x0  # tmc x0

.section .text
.type init_regs, @function
.local init_regs
init_regs:
#ifdef VX_CFG_VM_ENABLE
  # SATP is per-core, but each warp on each core writes the same value
  # idempotently — keeps the non-KMU init path symmetric with _start so
  # spawned warps don't rely on warp-0's earlier write being visible
  # before any user-VA access they make.
#if VX_CFG_VM_ADDR_MODE == SV39
  li    t0, VX_CFG_PAGE_TABLE_BASE_ADDR
  srli  t0, t0, VX_CFG_MEM_PAGE_LOG2_SIZE
  li    t1, 1
  slli  t1, t1, 63
  or    t0, t0, t1
  csrw  satp, t0
#elif VX_CFG_VM_ADDR_MODE == SV32
  li    t0, VX_CFG_PAGE_TABLE_BASE_ADDR
  srli  t0, t0, VX_CFG_MEM_PAGE_LOG2_SIZE
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
  LOAD_IMMEDIATE64(sp, VX_CFG_STACK_BASE_ADDR)
  csrr  t0, VX_CSR_MHARTID
  sll   t1, t0, VX_CFG_STACK_LOG2_SIZE
  sub   sp, sp, t1

#ifdef NEED_TLS
  # set thread pointer register: per-hart TLS image after the BSS region.
  # Stride is __tls_block_size (__tbss_offset + __tbss_size), not
  # __tbss_size alone — see the __vx_cta_entry TLS setup above.
  lui   t1, %hi(__tls_block_size)
  addi  t1, t1, %lo(__tls_block_size)
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
