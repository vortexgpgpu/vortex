#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t total_pages = 256;
uint32_t phys_words = 64;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h phys_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex VM stress test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n pages] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      total_pages = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer)  vx_buffer_release(src_buffer);
    if (dst_buffer)  vx_buffer_release(dst_buffer);
    if (phys_buffer) vx_buffer_release(phys_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_tasks = num_cores * num_warps * num_threads;
  uint32_t pages_per_task = (total_pages + num_tasks - 1) / num_tasks;
  uint32_t num_words = total_pages * WORDS_PER_PAGE;
  uint64_t buf_size = uint64_t(num_words) * sizeof(uint32_t);

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.pages_per_task = pages_per_task;
  kernel_arg.total_pages = total_pages;
  // odd stride so consecutive pages land in different TLB banks
  kernel_arg.stride_pages = 17;
  kernel_arg.phys_words = phys_words;

  std::cout << "pages: " << total_pages << ", tasks: " << num_tasks
            << ", pages/task: " << pages_per_task << std::endl;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  // physical (identity-mapped) buffer. Allocate, release, and allocate
  // again first: the freed slab range is typically handed back for the
  // second allocation, so the identity map is re-installed over the same
  // PA — the re-map must be idempotent and quiet.
  RT_CHECK(vx_buffer_create(device, phys_words * sizeof(uint32_t),
                            VX_MEM_READ | VX_MEM_PHYS, &phys_buffer));
  RT_CHECK(vx_buffer_release(phys_buffer));
  phys_buffer = nullptr;
  RT_CHECK(vx_buffer_create(device, phys_words * sizeof(uint32_t),
                            VX_MEM_READ | VX_MEM_PHYS, &phys_buffer));
  RT_CHECK(vx_buffer_address(phys_buffer, &kernel_arg.phys_addr));

  std::cout << "upload buffers" << std::endl;
  std::vector<uint32_t> h_src(num_words);
  std::vector<uint32_t> h_dst(num_words, 0);
  std::vector<uint32_t> h_phys(phys_words);
  for (uint32_t i = 0; i < num_words; ++i) {
    h_src[i] = i * 2654435761u;
  }
  for (uint32_t i = 0; i < phys_words; ++i) {
    h_phys[i] = 7 + i;
  }
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, dst_buffer, 0, h_dst.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, phys_buffer, 0, h_phys.data(), phys_words * sizeof(uint32_t), 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "launch kernel" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = num_tasks / num_threads;
    li.block_dim[0] = num_threads;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t t = 0; t < num_tasks; ++t) {
    uint32_t bias = h_phys[t % phys_words];
    for (uint32_t k = 0; k < pages_per_task; ++k) {
      uint32_t page = ((t * pages_per_task + k) * kernel_arg.stride_pages) % total_pages;
      uint32_t word = page * WORDS_PER_PAGE + (t % WORDS_PER_PAGE);
      uint32_t ref = h_src[word] + bias;
      if (h_dst[word] != ref) {
        if (errors < 20) {
          printf("*** error: task=%u page=%u word=%u expected=0x%x actual=0x%x\n",
                 t, page, word, ref, h_dst[word]);
        }
        ++errors;
      }
    }
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
