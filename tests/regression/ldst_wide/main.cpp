// ldst_wide — the wide loads and stores (vx_flw.d/.q, vx_fsw.d/.q) move what
// the scalar accesses would, in local and global memory, across lane patterns,
// register groups, word alignments and the limits of their offset.

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
  uint64_t isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,      &nc));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,      &isa));
  // without the wide forms the kernel moves the same words with generic accesses
  const bool wide = (isa & VX_ISA_EXT_LDST_WIDE) != 0;
  const uint32_t block = static_cast<uint32_t>(nt * nw);
  const uint32_t grid = static_cast<uint32_t>(nc * 2);
  const uint32_t num_threads = block * grid;
  const uint32_t words = RO_WORDS + block * SLOT_WORDS;
  std::cout << "ldst_wide: block=" << block << " grid=" << grid << " words=" << words
            << " wide=" << wide << std::endl;
  if (words * sizeof(uint32_t) > lmem_size) {
    std::cout << "local memory too small\nFAILED!" << std::endl;
    return 1;
  }

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  vx_buffer_h src_buf = nullptr;
  vx_buffer_h dst_buf = nullptr;
  vx_buffer_h err_buf = nullptr;
  vx_buffer_h dump_buf = nullptr;
  vx_buffer_h wide_buf = nullptr;
  CHECK(vx_buffer_create(dev, RO_WORDS * sizeof(uint32_t), VX_MEM_READ, &src_buf));
  CHECK(vx_buffer_create(dev, num_threads * SLOT_WORDS * sizeof(uint32_t), VX_MEM_READ_WRITE, &dst_buf));
  CHECK(vx_buffer_create(dev, num_threads * sizeof(uint32_t), VX_MEM_WRITE, &err_buf));
  CHECK(vx_buffer_create(dev, grid * words * sizeof(uint32_t), VX_MEM_WRITE, &dump_buf));
  CHECK(vx_buffer_create(dev, sizeof(uint32_t), VX_MEM_WRITE, &wide_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t kernel_arg{};
  kernel_arg.lmem_words  = words;
  CHECK(vx_buffer_address(src_buf,  &kernel_arg.src_addr));
  CHECK(vx_buffer_address(dst_buf,  &kernel_arg.dst_addr));
  CHECK(vx_buffer_address(err_buf,  &kernel_arg.err_addr));
  CHECK(vx_buffer_address(dump_buf, &kernel_arg.dump_addr));
  CHECK(vx_buffer_address(wide_buf, &kernel_arg.wide_addr));

  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = kern;
  li.args_host   = &kernel_arg;
  li.args_size   = sizeof(kernel_arg);
  li.ndim        = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0]= block;
  li.lmem_size   = words * sizeof(uint32_t);

  std::vector<uint32_t> h_src(RO_WORDS);
  for (uint32_t i = 0; i < RO_WORDS; ++i) {
    h_src[i] = ro_word(i) ^ GLOBAL_SALT;
  }
  std::vector<uint32_t> h_dst(num_threads * SLOT_WORDS, 0);
  std::vector<uint32_t> h_err(num_threads);
  std::vector<uint32_t> h_dump(grid * words);
  uint32_t h_wide = ~0u;
  vx_event_h write_ev[2] = {nullptr, nullptr};
  vx_event_h launch_ev = nullptr;
  vx_event_h read_ev = nullptr;
  vx_event_h dst_ev = nullptr;
  CHECK(vx_enqueue_write(q, src_buf, 0, h_src.data(), h_src.size() * sizeof(uint32_t), 0, nullptr, &write_ev[0]));
  CHECK(vx_enqueue_write(q, dst_buf, 0, h_dst.data(), h_dst.size() * sizeof(uint32_t), 0, nullptr, &write_ev[1]));
  CHECK(vx_enqueue_launch(q, &li, 2, write_ev, &launch_ev));
  CHECK(vx_enqueue_read(q, h_err.data(), err_buf, 0, num_threads * sizeof(uint32_t), 1, &launch_ev, nullptr));
  CHECK(vx_enqueue_read(q, &h_wide, wide_buf, 0, sizeof(uint32_t), 1, &launch_ev, nullptr));
  CHECK(vx_enqueue_read(q, h_dump.data(), dump_buf, 0, h_dump.size() * sizeof(uint32_t), 1, &launch_ev, &read_ev));
  CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, h_dst.size() * sizeof(uint32_t), 1, &launch_ev, &dst_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(dst_ev, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  // the kernel must take the wide path exactly where the ISA has it
  if (h_wide != (wide ? 1u : 0u)) {
    std::printf("*** kernel wide path=%u, expected %u\n", h_wide, wide ? 1u : 0u);
    ++errors;
  }
  for (uint32_t t = 0; t < num_threads; ++t) {
    if (h_err[t] != 0) {
      if (errors < 16) {
        std::printf("*** thread %u: %u wrong values\n", t, h_err[t]);
      }
      errors += h_err[t];
    }
  }
  for (uint32_t b = 0; b < grid; ++b) {
    for (uint32_t i = 0; i < words; ++i) {
      const uint32_t lid = (i - RO_WORDS) / SLOT_WORDS;
      uint32_t expected = (i < RO_WORDS) ? ro_word(i)
                        : slot_word(b * block + lid, lid, (i - RO_WORDS) % SLOT_WORDS);
      uint32_t actual = h_dump[b * words + i];
      if (actual != expected) {
        if (errors < 16) {
          std::printf("*** block %u lmem[%u] expected=0x%x actual=0x%x\n", b, i, expected, actual);
        }
        ++errors;
      }
    }
  }

  for (uint32_t t = 0; t < num_threads; ++t) {
    for (uint32_t k = 0; k < SLOT_WORDS; ++k) {
      uint32_t expected = slot_word(t, t % block, k);
      uint32_t actual = h_dst[t * SLOT_WORDS + k];
      if (actual != expected) {
        if (errors < 16) {
          std::printf("*** thread %u global slot[%u] expected=0x%x actual=0x%x\n", t, k, expected, actual);
        }
        ++errors;
      }
    }
  }

  vx_event_release(dst_ev);
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_event_release(write_ev[1]);
  vx_event_release(write_ev[0]);
  vx_buffer_release(wide_buf);
  vx_buffer_release(dump_buf);
  vx_buffer_release(err_buf);
  vx_buffer_release(dst_buf);
  vx_buffer_release(src_buf);
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
