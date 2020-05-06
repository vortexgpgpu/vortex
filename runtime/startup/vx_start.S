#include "../config.h"

.section .init, "ax"
.global _start
.type   _start, @function
_start:
  la a1, vx_set_sp
  li a0, NUM_WARPS    # activate all warps
  .word 0x00b5106b # wspawn a0(numWarps), a1(PC SPAWN)
  jal vx_set_sp
  li a0, 1
  .word 0x0005006b    # back to single thread
  # Initialize global pointerp
  # call __cxx_global_var_init
  # Clear the bss segment
  la      a0, _edata
  la      a2, _end
  sub     a2, a2, a0
  li      a1, 0
  call    memset
  la      a0, __libc_fini_array   # Register global termination functions
  call    atexit                  # to be called upon exit
  call    __libc_init_array       # Run global initialization functions
  call    main
  tail    exit
.size  _start, .-_start

.section .text
.type _exit, @function
.global _exit
_exit:
  li a0, 0
  .word 0x0005006b    # disable all threads


.section .text
.type vx_set_sp, @function
.global vx_set_sp
vx_set_sp:
  li a0, NUM_THREADS
  .word 0x0005006b    # activate all threads
  
  .option push
  .option norelax
  1:auipc gp, %pcrel_hi(__global_pointer$)
    addi  gp, gp, %pcrel_lo(1b)
  .option pop

  csrr a1, CSR_GTID    # get gtid
  slli a1, a1, 10      # multiply tid by 1024
  csrr a2, CSR_LTID    # get tid  
  slli a2, a2, 2       # multiply tid by 4
  lui  sp, STACK_BASE_ADDR # load base sp
  sub  sp, sp, a1      # sub thread block
  add  sp, sp, a2      # reduce addr collision for perf

  csrr a3, CSR_LWID    # get wid
  beqz a3, RETURN
  li a0, 0
  .word 0x0005006b     # tmc 0
RETURN:
  ret

.section .data
	.global __dso_handle
	.weak __dso_handle
__dso_handle:
	.long	0
  