
.equ A_WARPS, 7
.equ SIZE, 50

.section .text

.type queue_initialize, @function
.global queue_initialize
queue_initialize:
	mv t0, a0       # loading base address of q
	li t1, 0       # to initialize variables
	li t2, A_WARPS # Num of available warps
	sw t1, 0 (t0)  # start_i
	sw t1, 4 (t0)  # end_i
	sw t1, 8 (t0)  # num_j
	sw t2, 12(t0)  # total_warps
	sw t1, 16(t0)  # active_warps
	ret




.type queue_enqueue, @function
.global queue_enqueue
queue_enqueue:
	mv   t0, a0       # loding base address of q
	lw   t1, 8 (t0)  # t1 = num_j
	addi t1, t1, 1   # ++t1
	sw   t1, 8 (t0)  # num_j = t1
	addi t1, t0, 20  # t1 = jobs_addr
	lw   t4, 4 (t0)  # t4 = end_i
	slli t2, t4, 5   # index * 32 [log(sizeof(job))]
	add  t1, t1, t2  # jobs + index
	lw   t3, 0 (a1)  # wid
	sw   t3, 0 (t1)  # 
	lw   t3, 4 (a1)  # n_threads
	sw   t3, 4 (t1)  # 
	lw   t3, 8 (a1)  # base_sp
	sw   t3, 8 (t1)  # 
	lw   t3, 12(a1)  # func_ptr
	sw   t3, 12(t1)  # 
	lw   t3, 16(a1)  # args
	sw   t3, 16(t1)  # 
	lw   t3, 20(a1)  # assigned_warp
	sw   t3, 20(t1)  # 
	addi t4, t4, 1   # end_i++
	li   t5, SIZE    # size
	bne  t4, t5, ec  # if ((q.end_i + 1) == SIZE)
	mv   t4, zero
ec:
	sw   t4, 4 (t0) # end_i
	ret


.type queue_dequeue, @function
.global queue_dequeue

queue_dequeue:
	mv   t0, a0       # loading base address of q
	lw   t1, 8 (t0)  # t1 = num_j
	addi t1, t1, -1  # --t1
	sw   t1, 8 (t0)  # num_j = t1
	addi t1, t0, 20  # t1 = jobs_addr
	lw   t4, 0 (t0)  # t4 = start_i
	li   t6, SIZE    # size
	mv   t5, t4      # t5 = start_i
	addi t5, t5, 1   # t5++
	bne  t5, t6, dc  # if ((q.start_i + 1) == SIZE)
	mv   t5, zero
dc:
	sw   t5, 0(t0)   # storing start_i
	slli t2, t4, 5   # index * 32 [log(sizeof(job))]
	add  t1, t1, t2  # jobs + index
	lw   t3, 0 (t1)  # wid
	sw   t3, 0 (a1)  # 
	lw   t3, 4 (t1)  # n_threads
	sw   t3, 4 (a1)  # 
	lw   t3, 8 (t1)  # base_sp
	sw   t3, 8 (a1)  # 
	lw   t3, 12(t1)  # func_ptr
	sw   t3, 12(a1)  # 
	lw   t3, 16(t1)  # args
	sw   t3, 16(a1)  # 
	lw   t3, 20(t1)  # assigned_warp
	sw   t3, 20(a1)  # 
	ret


.type queue_isFull, @function
.global queue_isFull
queue_isFull:
	mv   t0, a0       # loading base address of q
	lw   t1, 8 (t0)  # t1 = num_j
	mv   a0, zero    # ret_val = 0
	li   t3, SIZE    # t3 = SIZE
	bne  t3, t1, qf  # if (num_j == 1)
	addi a0, a0, 1   # ret_val = 1;
qf:
	ret



.type queue_isEmpty, @function
.global queue_isEmpty
queue_isEmpty:
	mv   t0, a0       # loading base address of q
	lw   t1, 8 (t0)  # t1 = num_j
	mv   a0, zero    # ret_val = 0
	mv   t3, zero    # t3 = 0
	bne  t3, t1, qe  # if (num_j == 0)
	addi a0, a0, 1   # ret_val = 1;
qe:
	ret


.type queue_availableWarps, @function
.global queue_availableWarps
queue_availableWarps:
	mv   t0, a0       # loading base address of q
	lw   t1, 12(t0)  # t1 = total_warps
	lw   t2, 16(t0)  # t2 = active_warps
	sltu a0, t2, t1
	ret
