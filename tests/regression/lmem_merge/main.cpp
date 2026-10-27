// lmem_merge — local-memory reads that share words must each get their own
// word, and keep each thread's order with its own stores.

#include <vortex2.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do { \
  vx_result_t _r = (expr); \
  if (_r != VX_SUCCESS) { \
    std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
          __FILE__, __LINE__, #expr, vx_result_string(_r)); \
    std::exit(1); \
  } \
} while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t    num_iters   = 16;

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
      case 'n': num_iters   = std::atoi(optarg); break;
      case 'k': kernel_file = optarg;            break;
      default:
        std::cout << "Usage: [-n iterations] [-k kernel] [-h]" << std::endl;
        std::exit(c == 'h' ? 0 : -1);
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);
  if (num_iters == 0) {
    num_iters = 1;
  }

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));

  // One block per core holding all its warps, two waves of blocks.
  uint64_t nt = 0, nw = 0, nc = 0, lmem_size = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,      &nc));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
  const uint32_t W = static_cast<uint32_t>(nt);
  const uint32_t block = static_cast<uint32_t>(nt * nw);
  const uint32_t grid = static_cast<uint32_t>(nc * 2);
  const uint32_t num_threads = block * grid;
  // shrink the iteration count until a block's RAW and WAR words fit
  while (num_iters > 1 && (RO_WORDS + 2 * nw * num_iters) * sizeof(uint32_t) > lmem_size) {
    num_iters /= 2;
  }
  const uint32_t words = RO_WORDS + 2 * static_cast<uint32_t>(nw) * num_iters;
  std::cout << "lmem_merge: iters=" << num_iters << " block=" << block << " grid=" << grid
       << " words=" << words << std::endl;
  if (words * sizeof(uint32_t) > lmem_size) {
    std::cout << "local memory too small\nFAILED!" << std::endl;
    return 1;
  }

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  vx_buffer_h err_buf = nullptr, dump_buf = nullptr;
  CHECK(vx_buffer_create(dev, num_threads * sizeof(uint32_t), VX_MEM_WRITE, &err_buf));
  CHECK(vx_buffer_create(dev, grid * words * sizeof(uint32_t), VX_MEM_WRITE, &dump_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t kernel_arg{};
  kernel_arg.num_iters  = num_iters;
  kernel_arg.warp_size  = W;
  kernel_arg.lmem_words = words;
  kernel_arg.raw_base   = RO_WORDS;
  kernel_arg.war_base   = RO_WORDS + static_cast<uint32_t>(nw) * num_iters;
  CHECK(vx_buffer_address(err_buf,  &kernel_arg.err_addr));
  CHECK(vx_buffer_address(dump_buf, &kernel_arg.dump_addr));

  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = kern;
  li.args_host   = &kernel_arg;
  li.args_size   = sizeof(kernel_arg);
  li.ndim        = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0]= block;
  li.lmem_size   = words * sizeof(uint32_t);

  std::vector<uint32_t> h_err(num_threads), h_dump(grid * words);
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, h_err.data(), err_buf, 0, num_threads * sizeof(uint32_t), 1, &launch_ev, nullptr));
  CHECK(vx_enqueue_read(q, h_dump.data(), dump_buf, 0, grid * words * sizeof(uint32_t), 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  for (uint32_t t = 0; t < num_threads; ++t) {
    if (h_err[t] == 0) {
      continue;
    }
    if (errors < 16) {
      std::printf("*** thread %u: %u wrong reads\n", t, h_err[t]);
    }
    errors += h_err[t];
  }
  // every RAW and WAR word was stored by its lane; the read-only words kept their value
  for (uint32_t b = 0; b < grid; ++b) {
    for (uint32_t i = 0; i < words; ++i) {
      uint32_t expected = (i < RO_WORDS) ? (i * 3 + 2) : (LMEM_MARK + i);
      uint32_t actual = h_dump[b * words + i];
      if (actual != expected) {
        if (errors < 16) {
          std::printf("*** block %u lmem[%u] expected=0x%x actual=0x%x\n", b, i, expected, actual);
        }
        ++errors;
      }
    }
  }

  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_buffer_release(dump_buf);
  vx_buffer_release(err_buf);
  vx_kernel_release(kern);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
