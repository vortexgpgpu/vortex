#ifndef VX_API_H
#define VX_API_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  char * printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
} context_t;

typedef void (*vx_spawn_kernel_cb) (
  const void * /* arg */,
	const context_t * /* context */,
	uint32_t /* group_x */,
	uint32_t /* group_y */,
	uint32_t /* group_z */
);

typedef void (*vx_spawn_tasks_cb)(int task_id, void *arg);

typedef void (*vx_serial_cb)(void *arg);

void vx_spawn_kernel(context_t * ctx, vx_spawn_kernel_cb callback, void * arg);

void vx_spawn_tasks(int num_tasks, vx_spawn_tasks_cb callback, void * arg);

void vx_serial(vx_serial_cb callback, void * arg);

#ifdef __cplusplus
}
#endif

#endif