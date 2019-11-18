

# .section .init, "ax"
# .global _start
# _start:
#     .cfi_startproc
#     .cfi_undefined ra
#     .option push
#     .option norelax
#     la gp, __global_pointer$
#     .option pop
#     la sp, __stack_top
#     add s0, sp, zero
#     jal zero, main
#     .cfi_endproc
    # .end

.section .init, "ax"
  .global _start
  .type   _start, @function
_start:
    # Initialize SP
    # la sp, __stack_top
    la a1, vx_set_sp
    li a0, 4
    .word 0x00b5106b # wspawn a0(numWarps), a1(PC SPAWN)
    jal vx_set_sp
    li a0, 1
    .word 0x0005006b    # tmc 1
    # Initialize global pointerp
     # call __cxx_global_var_init
      # Clear the bss segment
      la      a0, _edata
      la      a2, _end
      sub     a2, a2, a0
      li      a1, 0
      call    memset
      la      a0, __libc_fini_array   # Register global termination functions
      call    atexit                  #  to be called upon exit
      call    __libc_init_array       # Run global initialization functions
      li a0, 4
      .word 0x0005006b    # tmc 4
      call    main
      tail    exit
      .size  _start, .-_start

.section .text
.type vx_set_sp, @function
.global vx_set_sp
vx_set_sp:
      li a0, 4
      .word 0x0005006b    # tmc 4
      
      .option push
      .option norelax
      1:auipc gp, %pcrel_hi(__global_pointer$)
        addi  gp, gp, %pcrel_lo(1b)
      .option pop

      csrr a3, 0x21        # get wid
      slli a3, a3, 0x1a    # shift by wid
      csrr a2, 0x20        # get tid
      slli a1, a2, 10      # multiply tid by 1024
      slli a2, a2, 2       # multiply tid by 4
      lui  sp, 0x6ffff     # load base sp
      sub  sp, sp, a1      # sub sp - (1024*tid)
      sub  sp, sp, a3      # shoft per warp
      add  sp, sp, a2      # shift sp for better performance

      csrr a3, 0x21        # get wid
      beqz a3, RETURN
      li a0, 0
      .word 0x0005006b    # tmc 0
  RETURN:
      ret



