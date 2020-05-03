
#ifndef VX_INTRINSICS

#define VX_INTRINSICS

#ifdef __cplusplus
extern "C" {
#endif

// Spawn warps
void vx_wspawn(unsigned numWarps, unsigned PC_spawn);

// Set thread mask
void vx_tmc(unsigned numThreads);

// Warp Barrier
void vx_barrier(unsigned barriedID, unsigned numWarps);

// Split on a predicate
void vx_split(unsigned predicate);

// Join
void vx_join(void);

// Return the warp thread index
unsigned vx_thread_id(void);

// Return the core warp index
unsigned vx_warp_id(void);

// Return processsor unique thread id
unsigned vx_thread_gid(void);

// Return processsor unique warp id
unsigned vx_warp_gid(void);

// Return number cycles
unsigned vx_num_cycles(void);

// Return number instructions
unsigned vx_num_instrs(void);

#define __if(b) vx_split(b); \
                if (b) 

#define __else else

#define __endif vx_join();

#ifdef __cplusplus
}
#endif

#endif