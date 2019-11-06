
#ifndef VX_API_

#define VX_API_


typedef void (*func_t)(void *);

void vx_spawnWarps(unsigned numWarps, unsigned numThreads, func_t func_ptr , void * args);








#endif