
#ifndef VX_INTRINSICS

#define VX_INTRINSICS

#ifdef __cplusplus
extern "C" {
#endif

// Spawn warps
void vx_wspawn(int numWarps, int PC_spawn);

// Set thread mask
void vx_tmc(int numThreads);

// Warp Barrier
void vx_barrier(int barriedID, int numWarps);

// Split on a predicate
void vx_split(int predicate);

// Join
void vx_join();

// Return the warp's unique thread id 
int vx_thread_id();

// Return the core's unique warp id
int vx_warp_id();

// Return processsor unique core id
int vx_core_id();

// Return processsor global thread id
int vx_thread_gid();

// Return processsor global warp id
int vx_warp_gid();

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