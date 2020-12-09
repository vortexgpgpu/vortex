#include <VX_config.h>

.section .text

.type vx_wspawn, @function
.global vx_wspawn
vx_wspawn:
	.word 0x00b5106b # wspawn a0(num_warps), a1(func_ptr)
	ret

.type vx_tmc, @function
.global vx_tmc
vx_tmc:
	.word 0x0005006b    # tmc a0
	ret

.type vx_barrier, @function
.global vx_barrier
vx_barrier:
	.word 0x00b5406b # barrier a0(barrier_id), a1(num_warps)
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
	csrr a0, CSR_LWID
	ret

.type vx_warp_gid, @function
.global vx_warp_gid
vx_warp_gid:
	csrr a0, CSR_GWID
	ret

.type vx_thread_id, @function
.global vx_thread_id
vx_thread_id:
	csrr a0, CSR_LTID
	ret

.type vx_thread_gid, @function
.global vx_thread_gid
vx_thread_gid:
	csrr a0, CSR_GTID
	ret

.type vx_core_id, @function
.global vx_core_id
vx_core_id:
	csrr a0, CSR_GCID
	ret

.type vx_num_threads, @function
.global vx_num_threads
vx_num_threads:
    csrr a0, CSR_NT
    ret

.type vx_num_warps, @function
.global vx_num_warps
vx_num_warps:
    csrr a0, CSR_NW
    ret

.type vx_num_cores, @function
.global vx_num_cores
vx_num_cores:
    csrr a0, CSR_NC
    ret
	
.type vx_num_cycles, @function
.global vx_num_cycles
vx_num_cycles:
    csrr a0, CSR_MCYCLE
    ret
    
.type vx_num_instrs, @function
.global vx_num_instrs
vx_num_instrs:
    csrr a0, CSR_MINSTRET
    ret