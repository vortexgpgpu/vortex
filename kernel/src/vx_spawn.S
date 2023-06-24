#include <VX_config.h>
#include <VX_types.h>

.type vx_serial, @function
.global vx_serial
vx_serial:    
#if (XLEN == 64)
    addi sp, sp, -48
    sd   ra, 40(sp)
    sd   s4, 32(sp)
    sd   s3, 24(sp)
    sd   s2, 16(sp)
    sd   s1, 8(sp)
    sd   s0, 0(sp)
#else
    addi sp, sp, -24
    sw   ra, 20(sp)
    sw   s4, 16(sp)
    sw   s3, 12(sp)
    sw   s2, 8(sp)
    sw   s1, 4(sp)
    sw   s0, 0(sp)
#endif
    mv   s4, a0                 # s4 <- callback
    mv   s3, a1                 # s3 <- arg
    csrr s2, CSR_NUM_THREADS    # s2 <- NT
    csrr s1, CSR_THREAD_ID      # s1 <- tid    
    li	 s0, 0                  # s0 <- index
label_loop:
    sub	 t0, s0, s1
    seqz t1, t0                 # (index != tid)
    .insn r 0x0b, 2, 0, x0, t1, x0  # split t0
    bnez t0, label_join
    mv   a0, s3                 # a0 <- arg
    jalr s4                     # callback(arg)
label_join:
    .insn r 0x0b, 3, 0, x0, x0, x0  # join
    addi s0, s0, 1              # index++
    blt	 s0, s2, label_loop     # loop back
#if (XLEN == 64)
    ld   ra, 40(sp)
    ld   s4, 32(sp)
    ld   s3, 24(sp)
    ld   s2, 16(sp)
    ld   s1, 8(sp)
    ld   s0, 0(sp)
    addi sp, sp, 48
#else
    lw   ra, 20(sp)
    lw   s4, 16(sp)
    lw   s3, 12(sp)
    lw   s2, 8(sp)
    lw   s1, 4(sp)
    lw   s0, 0(sp)
    addi sp, sp, 24
#endif    
    ret
    