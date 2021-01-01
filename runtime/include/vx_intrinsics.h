#ifndef VX_INTRINSICS_H
#define VX_INTRINSICS_H

#ifdef __cplusplus
extern "C" {
#endif

// Spawn warps
void vx_wspawn(int num_warps, unsigned func_ptr);

// Set thread mask
void vx_tmc(int num_threads);

// Warp Barrier
void vx_barrier(int barried_id, int num_warps);

// Split on a predicate
void vx_split(int predicate);

// Join
void vx_join();

// Return active warp's thread id 
int vx_thread_id();

// Return active core's local thread id
int vx_thread_lid();

// Return processsor global thread id
int vx_thread_gid();

// Return active core's local warp id
int vx_warp_id();

// Return processsor's global warp id
int vx_warp_gid();

// Return processsor core id
int vx_core_id();

// Return the number of threads in a warp
int vx_num_threads();

// Return the number of warps in a core
int vx_num_warps();

// Return the number of cores in the processsor
int vx_num_cores();

// Return the number of cycles
int vx_num_cycles();

// Return the number of instructions
int vx_num_instrs();

#define __if(b) vx_split(b); \
                if (b) 

#define __else else

#define __endif vx_join();

#ifdef __cplusplus
}
#endif

#endif