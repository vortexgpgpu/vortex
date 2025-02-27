#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg)
{
  // vector add
  auto buf_a = reinterpret_cast<TYPE *>(arg->addr_a);
  auto buf_b = reinterpret_cast<TYPE *>(arg->addr_b);
  auto buf_dst = reinterpret_cast<TYPE *>(arg->addr_dst);

  buf_dst[task_id] = buf_a[task_id] + buf_b[task_id];
}

int main()
{
  // this *should* agree with the address in vx_link32.ld, and compatible with modern args configuration
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  return vx_spawn_threads(1, &arg->dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
  // vx_spawn_tasks(arg->dim, (vx_spawn_tasks_cb)kernel_body, arg);

  return 0;
}
