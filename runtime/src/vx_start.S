#include <VX_config.h>

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
  beqz a0, label_exit_next 
  mv gp, a0
  ecall;

label_exit_next:
  # dump performance CSRs
  call  vx_perf_dump

  # disable all threads in current warp
  li a0, 0
  .insn s 0x6b, 0, x0, 0(a0)  # tmc a0

.section .text
.type vx_set_sp, @function
.global vx_set_sp
vx_set_sp:
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
  csrr a2, CSR_LTID    # get local thread id
  #else
  csrr a2, CSR_GTID    # get global thread id
  #endif
  slli a1, a2, STACK_LOG2_SIZE
  sub  sp, sp, a1      # sub thread block

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
  