#include <VX_config.h>

.type vx_putchar, @function
.global vx_putchar
vx_putchar:
    la t0, print_addr
    lw t0, 0(t0)
    csrr t1, CSR_GTID
    slli t1, t1, 16
    or t1, t1, a0
    sw t1, 0(t0)
    ret

.section .data
print_addr:
    .word IO_BUS_ADDR_COUT 

