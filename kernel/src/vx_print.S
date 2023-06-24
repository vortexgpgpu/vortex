#include <VX_config.h>
#include <VX_types.h>

.type vx_putchar, @function
.global vx_putchar
vx_putchar:
    csrr t0, CSR_MHARTID
    andi t0, t0, %lo(IO_COUT_SIZE-1)
#if (XLEN == 64)
    li t1, (IO_COUT_ADDR >> 32)        
    slli t1, t1, 32                       
    li t2, (IO_COUT_ADDR & 0xffffffff) 
    or t1, t1, t2   
#else
    li t1, IO_COUT_ADDR
#endif
    add t0, t0, t1    
    sb a0, 0(t0)
    ret
