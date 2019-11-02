

.type _start, @function
.global _start
_start:
    # li a0, 4
    # .word 0x0005006b    # tmc a0
    # csrr	a1,0x20
    # slli a1, a1, 2

    # la a2, 0x20000000
    # add a2, a2, a1
    # sw a1, 0(a2)

    # la a2, 0x40000000
    # add a2, a2, a1
    # li a3, 5
    # sw a3, 0(a2)

    # la a2, 0x80000000
    # add a2, a2, a1
    # li a3, 7
    # sw a3, 0(a2)

    # la a2, 0x60000000
    # add a2, a2, a1
    # li a3, 7
    # sw a3, 0(a2)

    # la a2, 0x20000000
    # add a2, a2, a1
    # lw a4, 0(a2)
    # li a0, 0
    # .word 0x0005006b    # tmc a0
    ##########################
    # li a0, 4
    # .word 0x0005006b    # tmc 4
    # csrr  a1,0x20       # get tid
    # slli a1, a1, 10      # multiply tid by 1024
    # lui  sp, 0x6ffff    # load base sp
    # sub sp, sp, a1      # sub sp - (1024*tid)

    #     la a2, 0x20000000
    #     csrr  a3,0x20       # get tid
    #     slli a3, a3, 2
    #     add a2, a2, a3
    #     sw a3, 0(a2)
    #     lw a4, 0(a2)

    # li a0, 0
    # .word 0x0005006b    # tmc a0
	##########################
    li a0, 4
    .word 0x0005006b    # tmc 4
    csrr  a1,0x20       # get tid
    slli a1, a1, 10      # multiply tid by 1024
    lui  sp, 0x6ffff    # load base sp
    sub sp, sp, a1      # sub sp - (1024*tid)
    li a0, 4       
    .word 0x0005006b    # tmc 4
    jal  main
    li a0, 0
    .word 0x0005006b    # tmc a0



