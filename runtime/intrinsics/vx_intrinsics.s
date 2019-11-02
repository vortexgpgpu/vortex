


.section .text


.type vx_wspawn, @function
.global vx_wspawn
vx_wspawn:
	.word 0x00b5106b # wspawn a0(numWarps), a1(PC SPAWN)
	ret

.type vx_tmc, @function
.global vx_tmc
vx_tmc:
	.word 0x0005006b    # tmc a0
	ret


.type vx_barrier, @function
.global vx_barrier
vx_barrier:
	.word 0x00b5406b # barrier a0(barrier id), a1(numWarps)
	ret

.type vx_split, @function
.global vx_split
vx_split:
	.word 0x0005206b    # split a0
	ret

.type vx_join, @function
.global vx_join
vx_join:
	.word 0x0000306b    #join
	ret


.type vx_warpID, @function
.global vx_warpID
vx_warpID:
	csrr a0, 0x21 # read warp IDs
	ret


.type vx_threadID, @function
.global vx_threadID
vx_threadID:
	csrr a0, 0x20 # read thread IDs
	ret
	