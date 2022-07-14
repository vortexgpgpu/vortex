#include <VX_config.h>

.section .init, "ax"
.global _start
.type   _start, @function
_start:

  # initialize per-thread registers
  csrr a0, CSR_NW  # get num warps
  la a1, init_regs
  .insn s 0x6b, 1, a1, 0(a0)  # wspawn a0, a1
  jal init_regs
  # return back to single thread execution
  li a0, 1
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0

  # initialize TLS for all warps
  csrr a0, CSR_NW  # get num warps
  la a1, __init_tls
  .insn s 0x6b, 1, a1, 0(a0)  # wspawn a0, a1
  call __init_tls
  # return back to single thread execution
  li a0, 1
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0
  
  # clear BSS segment
  la      a0, _edata
  la      a2, _end
  sub     a2, a2, a0
  li      a1, 0
  call    memset

  # Initialize trap vector
  # a t0, trap_entry
  # csrw mtvec, t0

  # Register global termination functions
  la      a0, __libc_fini_array

  # to be called upon exit
  call    atexit

  # Run global initialization functions
  call    __libc_init_array

  # call main program routine
  call    main

  # call exit routine
  tail    exit
.size  _start, .-_start

.section .text
.type _exit, @function
.global _exit
_exit:
  mv s0, a0
  call vx_perf_dump 
  mv gp, s0
  li a0, 0
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0

.section .text
.type init_regs, @function
.global init_regs
init_regs:
  # activate all threads
  li a0, -1
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0
  
  # set global pointer register
  .option push
  .option norelax
  la gp, __global_pointer
  .option pop

  # allocate stack region for a threads on the processor 
  # set stack pointer
  li sp, SMEM_BASE_ADDR # load stack base address
  #if SM_ENABLE
  csrr a0, CSR_LTID    # get local thread id
  #else
  csrr a0, CSR_GTID    # get global thread id
  #endif
  sll  a1, a0, STACK_LOG2_SIZE
  sub  sp, sp, a1

  # set thread pointer register
  # use address space after BSS region
  # ensure cacheline alignment
  la      a1, __tcb_aligned_size
  mul     a0, a0, a1
  la      tp, _end + 63
  add     tp, tp, a0
  and     tp, tp, -64

  # disable active warps except warp0
  csrr a3, CSR_LWID    # get local wid
  beqz a3, RETURN
  li a0, 0
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0
RETURN:
  ret

.section .data
	.global __dso_handle
	.weak __dso_handle
__dso_handle:
	.long	0
  