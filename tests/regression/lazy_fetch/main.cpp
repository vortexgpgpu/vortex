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

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t rounds = 16;

vx_device_h device = nullptr;
vx_buffer_h buf_buffer = nullptr;
vx_buffer_h err_buffer = nullptr;
vx_buffer_h set_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void show_usage() {
  std::cout << "Vortex Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-n rounds] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      rounds = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (buf_buffer) vx_buffer_release(buf_buffer);
    if (err_buffer) vx_buffer_release(err_buffer);
    if (set_buffer) vx_buffer_release(set_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static int launch(const kernel_arg_t& arg, uint32_t grid, uint32_t block) {
  vx_event_h ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = grid;
  li.block_dim[0] = block;
  int ret = vx_enqueue_launch(queue, &li, 0, nullptr, &ev);
  if (ret != 0) {
    return ret;
  }
  ret = vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
  vx_event_release(ev);
  return ret;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (rounds == 0) {
    rounds = 1;
  }

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads, isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));

  // One block per core, holding all of its warps.
  uint32_t block = num_warps * num_threads;
  uint32_t grid  = num_cores;
  uint32_t total_threads = grid * block;
  uint32_t num_regions = rounds * total_threads * NUM_CASES;
  uint32_t num_words = num_regions * REGION_WORDS;
  uint32_t buf_size = num_words * sizeof(uint32_t);
  uint32_t err_size = 2 * total_threads * sizeof(uint32_t);
  uint32_t set_words = rounds * total_threads * SET_STRIDE_WORDS;
  uint32_t set_size = set_words * sizeof(uint32_t);

  kernel_arg_t arg = {};
  arg.rounds      = rounds;
  arg.num_threads = total_threads;
  arg.warp_size   = num_threads;
  arg.cross_warp  = (num_warps % 2) == 0;
  arg.amo         = (isa_flags & VX_ISA_STD_A) != 0;

  std::cout << "threads: " << total_threads << ", amo: " << arg.amo << ", regions: " << num_regions
            << ", buffer size: " << buf_size << " bytes" << std::endl;

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &buf_buffer));
  RT_CHECK(vx_buffer_address(buf_buffer, &arg.buf_addr));
  RT_CHECK(vx_buffer_create(device, err_size, VX_MEM_READ_WRITE, &err_buffer));
  RT_CHECK(vx_buffer_address(err_buffer, &arg.err_addr));
  RT_CHECK(vx_buffer_create(device, set_size, VX_MEM_READ_WRITE, &set_buffer));
  RT_CHECK(vx_buffer_address(set_buffer, &arg.set_addr));

  std::vector<uint8_t> h_buf(buf_size);
  for (uint32_t i = 0; i < buf_size; ++i) {
    h_buf[i] = pattern_byte(i);
  }
  std::vector<uint32_t> h_err(2 * total_threads, 0xdeadbeef);
  std::vector<uint32_t> h_set(set_words);
  for (uint32_t i = 0; i < set_words; ++i) {
    h_set[i] = pattern_word(i);
  }

  std::cout << "upload buffers" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, buf_buffer, 0, h_buf.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, err_buffer, 0, h_err.data(), err_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, set_buffer, 0, h_set.data(), set_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "launch kernel (exercise)" << std::endl;
  arg.phase = 0;
  RT_CHECK(launch(arg, grid, block));

  std::cout << "launch kernel (re-check)" << std::endl;
  arg.phase  = 1;
  arg.rotate = (num_cores > 1) ? block : num_threads;
  RT_CHECK(launch(arg, grid, block));

  std::cout << "download buffers" << std::endl;
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_buf.data(), buf_buffer, 0, buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_read(queue, h_set.data(), set_buffer, 0, set_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_read(queue, h_err.data(), err_buffer, 0, err_size, 0, nullptr, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);

  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t t = 0; t < 2 * total_threads; ++t) {
    if (h_err[t] != 0) {
      if (errors < 16) {
        std::cout << "error: thread " << (t % total_threads) << (t < total_threads ? " (exercise)" : " (re-check)")
                  << " reported 0x" << std::hex << h_err[t] << std::dec << " mismatches" << std::endl;
      }
      ++errors;
    }
  }
  auto words = reinterpret_cast<const uint32_t*>(h_buf.data());
  for (uint32_t region = 0; region < num_regions; ++region) {
    for (uint32_t w = 0; w < REGION_WORDS; ++w) {
      uint32_t ref = expected_word(region, w, &arg);
      uint32_t cur = words[region * REGION_WORDS + w];
      if (cur != ref) {
        if (errors < 16) {
          std::cout << "error: region " << region << " (case " << (region % NUM_CASES) << ") word " << w
                    << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::dec << std::endl;
        }
        ++errors;
      }
    }
  }

  for (uint32_t i = 0; i < set_words; ++i) {
    uint32_t ref = expected_set_word(i);
    if (h_set[i] != ref) {
      if (errors < 16) {
        std::cout << "error: same-set word " << i << std::hex << ": actual 0x" << h_set[i]
                  << ", expected 0x" << ref << std::dec << std::endl;
      }
      ++errors;
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
