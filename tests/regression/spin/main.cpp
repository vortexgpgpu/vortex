// Launches a kernel that livelocks (every thread spins forever) and then
// waits on it, which never returns. The caller is expected to SIGKILL this
// process while the GPU is actively looping, then verify that the next
// vx_device_open's reset recovers the device (see reset_acceptance.sh).
#include <cstdio>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret) break;                                     \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
    return -1;                                                \
  } while (false)

int main() {
  vx_device_h device = nullptr;
  vx_queue_h  queue = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  kernel_arg_t arg = {};

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  arg.num_tasks = 1;
  arg.task_size = 1;
  arg.dim_x = 1;

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel      = kernel;
  li.args_host   = &arg;
  li.args_size   = sizeof(arg);
  li.ndim        = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = 1;

  printf("launching livelocked kernel (this never returns)...\n");
  fflush(stdout);
  vx_event_h launch_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  // Blocks forever on the spinning kernel (vx_queue_flush only kicks the
  // queue; the event is what tracks the launch to completion).
  RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  printf("unexpected: livelocked kernel completed?!\n");
  return -1;
}
