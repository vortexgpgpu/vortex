#ifndef VX_API_H
#define VX_API_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*func_t)(void *);

void vx_spawn_warps(int num_warps, int num_threads, func_t func_ptr , void * args);

#ifdef __cplusplus
}
#endif

#endif