

.type vx_print_str, @function
.global vx_print_str
vx_print_str:
    addi sp, sp, -12
    sw   ra, 0(sp)
    sw   a1, 4(sp)
bl:
    lbu  a1,0(a0)
    beqz a1,be
    jal  vx_printc
    addi a0, a0, 1
    j bl
be:
    lw   ra, 0(sp)
    lw   a1, 4(sp)
    addi sp, sp, 12
    ret


.type vx_printc, @function
.global vx_printc
vx_printc:
    la a7, 0x00010000
    sw a1, 0(a7)
    ret



