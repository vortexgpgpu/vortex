


.section .text

.type _start, @function
.global _start
_start:
    lui  sp, 0x7ffff
    jal  main
    ecall

.type createThreads, @function
.global createThreads
createThreads:
    mv    s7 ,a3    # Moving x_ptr to s7
    mv    s8 ,a4    # Moving y_ptr to s8
    mv    s9 ,a5    # Moving z_ptr to s9
    mv    s10,a6    # Moving assigned_warp to s10
    mv    t5 ,sp    # Saving the current stack pointer to t5
    mv    t2 , a0   # t2 = num_threads
loop_init:
    li    a0,1     # i = 0
loop_cond:
    bge   a0, t2, loop_done # i < num_threads
loop_body:
    addi  sp,sp,-2048 # Allocate 2k stack for new thread
    mv    t1, a0      # #lane = i
    .word 0x3506b           # clone register state
loop_inc:
    addi a0, a0, 1
    j loop_cond
loop_done:
    mv    sp,t5  # Restoring the stack
    li    a0,0   # setting tid = 0 for main thread
    mv    t6,a2  # setting func_addr 
    mv    s11,t2 # setting num_threads to spawn
    .word 0x1bfe0eb
    la     a0, reschedule_warps
    .word 0x5406b

.type printc, @function
.global printc
printc:
    la a7, 0x00010000
    sw a1, 0(a7)
    ret


.type wspawn, @function
.global wspawn
wspawn:
    la t1, createThreads
    .word 0x3006b  # WSPAWN instruction
    ret

.type print_consol, @function
.global print_consol
print_consol:
    addi sp, sp, -12
    sw   ra, 0(sp)
    sw   a1, 4(sp)
bl:
    lbu  a1,0(a0)
    beqz a1,be
    jal  printc
    addi a0, a0, 1
    j bl
be:
    lw   ra, 0(sp)
    lw   a1, 4(sp)
    addi sp, sp, 12
    ret

.type print_int, @function
.global print_int
print_int:
    addi sp, sp, -12
    sw   ra, 0(sp)
    sw   a1, 4(sp)
    addi a1, a0, 48
    jal  printc
    lw   ra, 0(sp)
    lw   a1, 4(sp)
    addi sp, sp, 12
    ret
