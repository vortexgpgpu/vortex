


.section .text

.type _start, @function
.global _start
_start:
    lui  sp, 0x7ffff
    jal  vx_before_main
    jal  main
    ecall

.type vx_createThreads, @function
.global vx_createThreads
vx_createThreads:
    mv    s7 ,a3    # Moving args to s7
    mv    s10,a4    # Moving assigned_warp to s10
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
    la     a0, vx_reschedule_warps
    .word 0x5406b


.type vx_wspawn, @function
.global vx_wspawn
vx_wspawn:
    la t1, vx_createThreads
    .word 0x3006b  # WSPAWN instruction
    ret

