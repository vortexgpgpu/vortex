#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret)                                            \
      break;                                                  \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
    cleanup();                                                \
    exit(-1);                                                 \
  } while (false)

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h ro_buffer = nullptr;
kernel_arg_t kernel_arg = {};
uint32_t test_mode = FAULT_MODE_UNMAPPED;

void cleanup() {
  if (device) {
    if (dst_buffer) {
      vx_buffer_release(dst_buffer);
    }
    if (ro_buffer) {
      vx_buffer_release(ro_buffer);
    }
    if (kernel) {
      vx_kernel_release(kernel);
    }
    if (module_) {
      vx_module_release(module_);
    }
    if (queue) {
      vx_queue_release(queue);
    }
    vx_device_release(device);
  }
}

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:h")) != -1) {
    switch (c) {
    case 't': test_mode = (uint32_t)atoi(optarg); break;
    default:
      std::cout << "Usage: [-t mode]  (0=unmapped, 1=read-only store)"
                << std::endl;
      exit(c == 'h' ? 0 : -1);
    }
  }

  RT_CHECK(vx_device_open(0, &device));

  uint64_t vm_enabled = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_VM_SUPPORT, &vm_enabled));
  if (0 == vm_enabled) {
    // Without VM there is nothing to fault on; vacuous pass keeps the
    // case runnable in mixed suites.
    std::cout << "vm_fault: device has no VM support, skipping" << std::endl;
    std::cout << "PASSED!" << std::endl;
    cleanup();
    return 0;
  }

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_buffer_create(device, 4096, VX_MEM_READ_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  kernel_arg.mode = test_mode;
  if (test_mode == FAULT_MODE_READONLY) {
    // A read-only allocation: mapped and readable, but its leaf PTE
    // carries no write permission.
    RT_CHECK(vx_buffer_create(device, 4096, VX_MEM_READ, &ro_buffer));
    RT_CHECK(vx_buffer_address(ro_buffer, &kernel_arg.bad_addr));
  } else {
    // Below the identity-mapped low range and above the user heap, so it is
    // neither a live allocation nor a system mapping. It also has to stay
    // clear of every device aperture — an OM or IO address bypasses
    // translation entirely and would never fault.
    kernel_arg.bad_addr = uint64_t(VX_MEM_OM_BASE_ADDR) - 0x10000000ull;
    if (kernel_arg.bad_addr < VX_MEM_IO_END_ADDR) {
      std::cout << "vm_fault: no unmapped range available, skipping" << std::endl;
      std::cout << "PASSED!" << std::endl;
      cleanup();
      return 0;
    }
  }

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &kernel_arg;
  li.args_size    = sizeof(kernel_arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = 1;

  int err = vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev);
  if (err == 0) {
    err = vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(launch_ev);
  }

  cleanup();

  if (err == 0) {
    std::cout << "FAILED! - unmapped access did not raise a device error"
              << std::endl;
    return -1;
  }
  std::cout << "device error observed as expected (" << err << ")" << std::endl;
  std::cout << "PASSED!" << std::endl;
  return 0;
}
