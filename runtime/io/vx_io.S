#include "../config.h"

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
    la t0, print_addr
    lw t0, 0(t0)
    sw a1, 0(t0)
    ret

.section .data
print_addr:
    .word IO_BUS_ADDR_COUT 

