// lds_ldg — the explicit local (vx_lds/vx_sts) and global (vx_ldg/vx_stg)
// loads and stores return and store what the generic path does, at every
// width and at the limits of their offsets.

#include <vortex2.h>
#include "common.h"

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

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
      case 'k': kernel_file = optarg; break;
      default:
        std::cout << "Usage: [-k kernel] [-h]" << std::endl;
        std::exit(c == 'h' ? 0 : -1);
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));

  // One block per core holding all its warps, two waves of blocks.
  uint64_t nt = 0;
  uint64_t nw = 0;
  uint64_t nc = 0;
  uint64_t lmem_size = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,      &nc));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
  const uint32_t block = static_cast<uint32_t>(nt * nw);
  const uint32_t grid = static_cast<uint32_t>(nc * 2);
  const uint32_t num_threads = block * grid;
  const uint32_t words = RO_WORDS + block * SLOT_WORDS;
  std::cout << "lds_ldg: block=" << block << " grid=" << grid << " words=" << words << std::endl;
  if (words * sizeof(uint32_t) > lmem_size) {
    std::cout << "local memory too small\nFAILED!" << std::endl;
    return 1;
  }

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  const uint32_t in_words = num_threads + IN_PAD;
  std::vector<uint32_t> h_in(in_words);
  for (uint32_t i = 0; i < in_words; ++i) {
    h_in[i] = in_word(i);
  }

  vx_buffer_h in_buf = nullptr;
  vx_buffer_h out_buf = nullptr;
  vx_buffer_h err_buf = nullptr;
  vx_buffer_h dump_buf = nullptr;
  CHECK(vx_buffer_create(dev, in_words * sizeof(uint32_t), VX_MEM_READ, &in_buf));
  CHECK(vx_buffer_create(dev, num_threads * SLOT_WORDS * sizeof(uint32_t), VX_MEM_WRITE, &out_buf));
  CHECK(vx_buffer_create(dev, num_threads * sizeof(uint32_t), VX_MEM_WRITE, &err_buf));
  CHECK(vx_buffer_create(dev, grid * words * sizeof(uint32_t), VX_MEM_WRITE, &dump_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t kernel_arg{};
  kernel_arg.lmem_words  = words;
  CHECK(vx_buffer_address(in_buf,   &kernel_arg.in_addr));
  CHECK(vx_buffer_address(out_buf,  &kernel_arg.out_addr));
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

  std::vector<uint32_t> h_out(num_threads * SLOT_WORDS);
  std::vector<uint32_t> h_err(num_threads);
  std::vector<uint32_t> h_dump(grid * words);
  vx_event_h write_ev = nullptr;
  vx_event_h launch_ev = nullptr;
  vx_event_h read_ev = nullptr;
  CHECK(vx_enqueue_write(q, in_buf, 0, h_in.data(), in_words * sizeof(uint32_t), 0, nullptr, &write_ev));
  CHECK(vx_enqueue_launch(q, &li, 1, &write_ev, &launch_ev));
  CHECK(vx_enqueue_read(q, h_out.data(), out_buf, 0, h_out.size() * sizeof(uint32_t), 1, &launch_ev, nullptr));
  CHECK(vx_enqueue_read(q, h_err.data(), err_buf, 0, num_threads * sizeof(uint32_t), 1, &launch_ev, nullptr));
  CHECK(vx_enqueue_read(q, h_dump.data(), dump_buf, 0, h_dump.size() * sizeof(uint32_t), 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  for (uint32_t t = 0; t < num_threads; ++t) {
    if (h_err[t] != 0) {
      if (errors < 16) {
        std::printf("*** thread %u: %u wrong values\n", t, h_err[t]);
      }
      errors += h_err[t];
    }
    for (uint32_t k = 0; k < SLOT_WORDS; ++k) {
      uint32_t actual = h_out[t * SLOT_WORDS + k];
      if (actual != out_word(t, k)) {
        if (errors < 16) {
          std::printf("*** out[%u][%u] expected=0x%x actual=0x%x\n", t, k, out_word(t, k), actual);
        }
        ++errors;
      }
    }
  }
  for (uint32_t b = 0; b < grid; ++b) {
    for (uint32_t i = 0; i < words; ++i) {
      uint32_t expected = (i < RO_WORDS) ? ro_word(i)
                        : slot_word(b * block + (i - RO_WORDS) / SLOT_WORDS, (i - RO_WORDS) % SLOT_WORDS);
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
  vx_event_release(write_ev);
  vx_buffer_release(dump_buf);
  vx_buffer_release(err_buf);
  vx_buffer_release(out_buf);
  vx_buffer_release(in_buf);
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
