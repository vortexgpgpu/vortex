#include <VX_config.h>
#include <VX_types.h>

.section .init, "ax"
.global _start
.type   _start, @function
_start:
  
  # execute stack initialization on all warps
  la a1, vx_set_sp
  csrr a0, CSR_NW  # get num warps
  .insn s 0x6b, 1, a1, 0(a0)  # wspawn a0, a1
  jal vx_set_sp

  # return back to single thread execution
  li a0, 1
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0
  
  # Clear the bss segment
  la      a0, _edata
  la      a2, _end
  sub     a2, a2, a0
  li      a1, 0
  call    memset

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
.type vx_set_sp, @function
.global vx_set_sp
vx_set_sp:
  # activate all threads
  li a0, -1
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0
  
  # set per-thread stack register
  li sp, STACK_BASE_ADDR # load stack base address
  li a1, STACK_SIZE      # load stack size
  csrr a2, CSR_GTID      # get global thread id
  mul  a1, a1, a2
  sub  sp, sp, a1

  # disable active warps except warp0
  csrr a3, CSR_LWID     # get local wid
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
  