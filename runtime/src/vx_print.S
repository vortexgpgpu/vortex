#include <VX_config.h>

.type vx_putchar, @function
.global vx_putchar
vx_putchar:
    csrr t0, CSR_GTID
    andi t0, t0, %lo(IO_COUT_SIZE-1)
    li t1, IO_COUT_ADDR
    add t0, t0, t1    
    sb a0, 0(t0)
    ret