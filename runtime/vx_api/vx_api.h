
#ifndef VX_API_

#define VX_API_

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*func_t)(void *);

void vx_spawnWarps(unsigned numWarps, unsigned numThreads, func_t func_ptr , void * args);


void vx_cl_spawnWarps (char * /* args */,
					     char * /* pocl_context */,
					     long /* group_x */,
					     long /* group_y */,
					     long /* group_z */);

#ifdef __cplusplus
}
#endif


#endif