

.type _start, @function
.global _start
_start:
    # la a1, vx_set_sp
    # li a0, 8
    # .word 0x00b5106b # wspawn a0(numWarps), a1(PC SPAWN)
    jal vx_set_sp
    jal  main
    li a0, 0
    .word 0x0005006b    # tmc a0


.type vx_set_sp, @function
.global vx_set_sp
vx_set_sp:
    li a0, 4
    .word 0x0005006b    # tmc 4

    csrr a3, 0x21        # get wid
    slli a3, a3, 15      # shift by wid
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
    li a0, 1      
    .word 0x0005006b    # tmc 1
    ret



