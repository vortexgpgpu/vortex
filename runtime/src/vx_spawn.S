.type vx_serial, @function
.global vx_serial
vx_serial: 
    addi sp, sp, -24
    sw   ra, 20(sp)
    sw   s4, 16(sp)
    sw   s3, 12(sp)
    sw   s2, 8(sp)
    sw   s1, 4(sp)
    sw   s0, 0(sp)
    mv   s4, a0     # callback
    mv   s3, a1     # arg
    csrr s2, 0xfc0  # NT
    csrr s1, 0xcc0  # tid    
    li	 s0, 0      # index
label_loop:
    sub	 t0, s0, s1
    snez t0, t0
    .insn s 0x6b, 2, x0, 0(t0)  # split t0
    bnez t0, label_join
    mv   a0, s0     # a0 <- index
    mv   a1, s3     # a1 <- arg
    jalr s4         # callback(index, arg)
label_join:
    .insn s 0x6b, 3, x0, 0(x0)  # join
    addi s0, s0, 1
    blt	 s0, s2, label_loop
    lw   ra, 20(sp)
    lw   s4, 16(sp)
    lw   s3, 12(sp)
    lw   s2, 8(sp)
    lw   s1, 4(sp)
    lw   s0, 0(sp)
    addi sp, sp, 24

    ret