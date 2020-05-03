#include "../config.h"

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

.type vx_warp_id, @function
.global vx_warp_id
vx_warp_id:
	csrr a0, CSR_LWID # read warp index
	ret

.type vx_warp_gid, @function
.global vx_warp_gid
vx_warp_gid:
	csrr a0, CSR_GWID # read warp index
	ret

.type vx_thread_id, @function
.global vx_thread_id
vx_thread_id:
	csrr a0, CSR_LTID # read thread index
	ret

.type vx_thread_gid, @function
.global vx_thread_gid
vx_thread_gid:
	csrr a0, CSR_GTID # read thread index
	ret
	
.type vx_num_cycles, @function
.global vx_num_cycles
vx_num_cycles:
    csrr a0, CSR_CYCLL
    ret
    
.type vx_num_instrs, @function
.global vx_num_instrs
vx_num_instrs:
    csrr a0, CSR_INSTL
    ret