


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
	

.type vx_resetStack, @function
.global vx_resetStack
vx_resetStack:
    li a0, 4
    .word 0x0005006b    # tmc 4

    csrr a3, 0x21        # get wid
    slli a3, a3, 15      # shift by wid
    csrr a2, 0x20        # get tid
    slli a1, a2, 10      # multiply tid by 1024
    slli a2, a2, 2       # multiply tid by 4
    lui  sp, 0x6ffff     # load base sp
    sub  sp, sp, a1      # sub sp - (1024*tid)
    sub  sp, sp, a3      # shoft per warp
    add  sp, sp, a2      # shift sp for better performance

    csrr a3, 0x21        # get wid
    beqz a3, RETURN
    li a0, 0
    .word 0x0005006b    # tmc 0
RETURN:
    ret

