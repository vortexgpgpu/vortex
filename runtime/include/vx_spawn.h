#ifndef VX_API_H
#define VX_API_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  char * printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
};

typedef void (*pfn_workgroup_func) (
  const void * /* args */,
	const struct context_t * /* context */,
	uint32_t /* group_x */,
	uint32_t /* group_y */,
	uint32_t /* group_z */
);

typedef void (*pfn_callback)(int task_id, const void *arg);

void vx_spawn_kernel(struct context_t * ctx, pfn_workgroup_func wg_func, const void * args);

void vx_spawn_tasks(int num_tasks, pfn_callback callback, const void * args);

#ifdef __cplusplus
}
#endif

#endif