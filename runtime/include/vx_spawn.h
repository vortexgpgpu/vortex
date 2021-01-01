#ifndef VX_API_H
#define VX_API_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*pfn_callback)(int task_id, void *arg);

void vx_spawn_tasks(int num_tasks, pfn_callback callback , void * args);

#ifdef __cplusplus
}
#endif

#endif