
#ifndef VX_INTRINSICS

#define VX_INTRINSICS

#ifdef __cplusplus
extern "C" {
#endif

// Spawns Warps
void vx_wspawn (unsigned numWarps, unsigned PC_spawn);

// Changes thread mask (activated/deactivates threads)
void vx_tmc    (unsigned numThreads);

// Warp Barrier
void vx_barrier(unsigned barriedID, unsigned numWarps);

// split on a predicate
void vx_split  (unsigned predicate);


// Join
void vx_join   (void);


// Get Hardware thread ID
unsigned vx_threadID(void);


// Get hardware warp ID
unsigned vx_warpID(void);

void vx_resetStack(void);


#define __if(b) vx_split(b); \
                if (b) 

#define __else else


#define __endif vx_join();


#ifdef __cplusplus
}
#endif

#endif