#include <iostream>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#define PAGE_SIZE 4096u
#define WORDS_PER_PAGE (PAGE_SIZE / sizeof(uint32_t))

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
uint32_t test_mode = VM_MODE_STRIDE;
uint32_t num_tasks = 16;
uint32_t pages_per_task = 4;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
vx_buffer_h buf_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h aux_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-t mode] [-n tasks] [-p pages/task] [-h]" << std::endl;
  std::cout << "  modes: 0=stride 1=fence 2=drain 3=amo 4=superpage" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:n:p:h")) != -1) {
    switch (c) {
    case 't': test_mode = atoi(optarg); break;
    case 'n': num_tasks = atoi(optarg); break;
    case 'p': pages_per_task = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (buf_buffer) {
      vx_buffer_release(buf_buffer);
    }
    if (dst_buffer) {
      vx_buffer_release(dst_buffer);
    }
    if (aux_buffer) {
      vx_buffer_release(aux_buffer);
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
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  uint64_t buf_size = (uint64_t)num_tasks * pages_per_task * PAGE_SIZE;
  uint64_t dst_size = num_tasks * sizeof(uint32_t);
  uint64_t aux_size = pages_per_task * PAGE_SIZE;

  std::cout << "vm_test: mode=" << test_mode << ", tasks=" << num_tasks
            << ", pages/task=" << pages_per_task << std::endl;

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &buf_buffer));
  RT_CHECK(vx_buffer_create(device, dst_size, VX_MEM_READ_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_create(device, aux_size, VX_MEM_READ_WRITE, &aux_buffer));
  RT_CHECK(vx_buffer_address(buf_buffer, &kernel_arg.buf_addr));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  RT_CHECK(vx_buffer_address(aux_buffer, &kernel_arg.aux_addr));

  kernel_arg.mode = test_mode;
  kernel_arg.num_tasks = num_tasks;
  kernel_arg.pages_per_task = pages_per_task;

  // Host staging kept alive until the enqueued writes complete (the
  // queue worker reads them asynchronously).
  std::vector<uint8_t> zeros(aux_size, 0);
  std::vector<uint32_t> poison(buf_size / sizeof(uint32_t), 0xCAFED00Du);

  if (test_mode == VM_MODE_SUPERPAGE) {
    // Walk the page table through its own identity superpage mapping.
    kernel_arg.aux_addr = VX_MEM_PAGE_TABLE_BASE_ADDR;
  } else {
    RT_CHECK(vx_enqueue_write(queue, aux_buffer, 0, zeros.data(), aux_size, 0, nullptr, nullptr));
  }

  // Poison the working buffer so untouched pages are detectable.
  RT_CHECK(vx_enqueue_write(queue, buf_buffer, 0, poison.data(), buf_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  uint32_t block_x = std::min(num_tasks, 64u);
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = (num_tasks + block_x - 1) / block_x;
    li.block_dim[0] = block_x;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::vector<uint32_t> dst(num_tasks);
  RT_CHECK(vx_enqueue_read(queue, dst.data(), dst_buffer, 0, dst_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  switch (test_mode) {
  case VM_MODE_STRIDE: {
    for (uint32_t g = 0; g < num_tasks; ++g) {
      uint32_t sum = 0;
      for (uint32_t p = 0; p < pages_per_task; ++p) {
        sum += g ^ (0x1234567u + p);
      }
      if (dst[g] != sum) {
        if (errors < 16) {
          printf("*** error: [%u] expected=0x%x, actual=0x%x\n", g, sum, dst[g]);
        }
        ++errors;
      }
    }
    break;
  }
  case VM_MODE_FENCE: {
    for (uint32_t g = 0; g < num_tasks; ++g) {
      uint32_t peer = (g + 1) % num_tasks;
      uint32_t expected = 0xAB000000u + peer;
      if (dst[g] != expected) {
        if (errors < 16) {
          printf("*** error: [%u] expected=0x%x, actual=0x%x\n", g, expected, dst[g]);
        }
        ++errors;
      }
    }
    break;
  }
  case VM_MODE_DRAIN: {
    // The trailing stores must be visible to the host readback with no
    // fence: read the working buffer directly.
    std::vector<uint32_t> data(buf_size / sizeof(uint32_t));
    vx_event_h drain_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, data.data(), buf_buffer, 0, buf_size, 0, nullptr, &drain_ev));
    RT_CHECK(vx_event_wait_value(drain_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(drain_ev);
    for (uint32_t g = 0; g < num_tasks; ++g) {
      for (uint32_t p = 0; p < pages_per_task; ++p) {
        uint32_t idx = (g * pages_per_task + p) * WORDS_PER_PAGE;
        uint32_t expected = 0xD0000000u + g * pages_per_task + p;
        if (data[idx] != expected) {
          if (errors < 16) {
            printf("*** error: [%u,%u] expected=0x%x, actual=0x%x\n",
                   g, p, expected, data[idx]);
          }
          ++errors;
        }
      }
    }
    break;
  }
  case VM_MODE_AMO: {
    std::vector<uint32_t> ctrs(aux_size / sizeof(uint32_t));
    vx_event_h amo_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, ctrs.data(), aux_buffer, 0, aux_size, 0, nullptr, &amo_ev));
    RT_CHECK(vx_event_wait_value(amo_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(amo_ev);
    uint32_t expected = num_tasks * (num_tasks + 1) / 2;
    for (uint32_t p = 0; p < pages_per_task; ++p) {
      uint32_t v = ctrs[p * WORDS_PER_PAGE];
      if (v != expected) {
        if (errors < 16) {
          printf("*** error: ctr[%u] expected=%u, actual=%u\n", p, expected, v);
        }
        ++errors;
      }
    }
    break;
  }
  case VM_MODE_SUPERPAGE: {
    for (uint32_t g = 0; g < num_tasks; ++g) {
      if (dst[g] != 1u) {
        if (errors < 16) {
          printf("*** error: [%u] superpage probe failed (0x%x)\n", g, dst[g]);
        }
        ++errors;
      }
    }
    break;
  }
  default:
    break;
  }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED! - " << errors << " errors" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
