


.section .text

.type _start, @function
.global _start
_start:
    li a0, 4          # Num Warps
    csrw 0x20, a0     # Setting the number of available warps 
    li a0, 8          # Num Threads
    csrw 0x21, a0     # Setting the number of available threads
    csrw mhartid,zero
    csrw misa,zero
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

.global context

.type vx_save_context, @function
.global vx_save_context
vx_save_context:
la tp, context
sw x0 , 0 (tp)
sw x1 , 4 (tp)
sw x2 , 8 (tp)
sw x3 , 12(tp)
sw x4 , 16(tp)
sw x5 , 20(tp)
sw x6 , 24(tp)
sw x7 , 28(tp)
sw x8 , 32(tp)
sw x9 , 36(tp)
sw x10, 40(tp)
sw x11, 44(tp)
sw x12, 48(tp)
sw x13, 52(tp)
sw x14, 56(tp)
sw x15, 60(tp)
sw x16, 64(tp)
sw x17, 68(tp)
sw x18, 72(tp)
sw x19, 76(tp)
sw x20, 80(tp)
sw x21, 84(tp)
sw x22, 88(tp)
sw x23, 92(tp)
sw x24, 96(tp)
sw x25, 100(tp)
sw x26, 104(tp)
sw x27, 108(tp)
sw x28, 112(tp)
sw x29, 116(tp)
sw x30, 120(tp)
sw x31, 124(tp)
li tp, 1
ret


.type vx_load_context, @function
.global vx_load_context
vx_load_context:
la tp, context
lw x0 , 0 (tp)
lw x1 , 4 (tp)
lw x2 , 8 (tp)
lw x3 , 12(tp)
lw x4 , 16(tp)
lw x5 , 20(tp)
lw x6 , 24(tp)
lw x7 , 28(tp)
lw x8 , 32(tp)
lw x9 , 36(tp)
lw x10, 40(tp)
lw x11, 44(tp)
lw x12, 48(tp)
lw x13, 52(tp)
lw x14, 56(tp)
lw x15, 60(tp)
lw x16, 64(tp)
lw x17, 68(tp)
lw x18, 72(tp)
lw x19, 76(tp)
lw x20, 80(tp)
lw x21, 84(tp)
lw x22, 88(tp)
lw x23, 92(tp)
lw x24, 96(tp)
lw x25, 100(tp)
lw x26, 104(tp)
lw x27, 108(tp)
lw x28, 112(tp)
lw x29, 116(tp)
lw x30, 120(tp)
lw x31, 124(tp)
li tp, 0
ret

.type vx_available_warps, @function
.global vx_available_warps
vx_available_warps:
csrr a0, 0x20
ret

.type vx_available_threads, @function
.global vx_available_threads
vx_available_threads:
csrr a0, 0x21
ret





