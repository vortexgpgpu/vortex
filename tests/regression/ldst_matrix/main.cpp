// ldst_matrix — the matrix fragment loads and stores (vx_ldm, vx_stm) move the
// words of each fragment shape the configuration has, from and to local and
// global memory, with row strides that are not powers of two, zero, positive
// and negative offsets, and aligned and unaligned register bases.

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

  uint64_t nt = 0;
  uint64_t nc = 0;
  uint64_t lmem_size = 0;
  uint64_t isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,      &nc));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,      &isa));
  if (!(isa & VX_ISA_EXT_LDST_MATRIX)) {
    std::cout << "matrix load/store extension not supported!\nFAILED!" << std::endl;
    return 1;
  }

  // a block is one warp, as far as each thread's two local regions fit
  const uint32_t lmem_per_thread = 2 * LOCAL_REGION_WORDS * sizeof(uint32_t);
  const uint32_t block = static_cast<uint32_t>(std::min<uint64_t>(nt, lmem_size / lmem_per_thread));
  const uint32_t grid = static_cast<uint32_t>(nc * 2);
  const uint32_t num_threads = block * grid;
  const uint32_t global_words = num_threads * GLOBAL_REGION_WORDS;
  std::cout << "ldst_matrix: block=" << block << " grid=" << grid << std::endl;

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  vx_buffer_h src_buf = nullptr;
  vx_buffer_h dst_buf = nullptr;
  vx_buffer_h err_buf = nullptr;
  vx_buffer_h shape_buf = nullptr;
  CHECK(vx_buffer_create(dev, global_words * sizeof(uint32_t), VX_MEM_READ, &src_buf));
  CHECK(vx_buffer_create(dev, global_words * sizeof(uint32_t), VX_MEM_READ_WRITE, &dst_buf));
  CHECK(vx_buffer_create(dev, num_threads * sizeof(uint32_t), VX_MEM_WRITE, &err_buf));
  CHECK(vx_buffer_create(dev, sizeof(uint32_t), VX_MEM_WRITE, &shape_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t kernel_arg{};
  CHECK(vx_buffer_address(src_buf,   &kernel_arg.src_addr));
  CHECK(vx_buffer_address(dst_buf,   &kernel_arg.dst_addr));
  CHECK(vx_buffer_address(err_buf,   &kernel_arg.err_addr));
  CHECK(vx_buffer_address(shape_buf, &kernel_arg.shape_addr));

  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = kern;
  li.args_host   = &kernel_arg;
  li.args_size   = sizeof(kernel_arg);
  li.ndim        = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0]= block;
  li.lmem_size   = block * lmem_per_thread;

  std::vector<uint32_t> h_src(global_words);
  for (uint32_t t = 0; t < num_threads; ++t) {
    for (uint32_t i = 0; i < GLOBAL_REGION_WORDS; ++i) {
      h_src[t * GLOBAL_REGION_WORDS + i] = src_word(t, i, GLOBAL_SALT);
    }
  }
  std::vector<uint32_t> h_dst(global_words, 0);
  std::vector<uint32_t> h_err(num_threads);
  uint32_t h_shapes = 0;
  vx_event_h write_ev[2] = {nullptr, nullptr};
  vx_event_h launch_ev = nullptr;
  vx_event_h read_ev = nullptr;
  CHECK(vx_enqueue_write(q, src_buf, 0, h_src.data(), h_src.size() * sizeof(uint32_t), 0, nullptr, &write_ev[0]));
  CHECK(vx_enqueue_write(q, dst_buf, 0, h_dst.data(), h_dst.size() * sizeof(uint32_t), 0, nullptr, &write_ev[1]));
  CHECK(vx_enqueue_launch(q, &li, 2, write_ev, &launch_ev));
  CHECK(vx_enqueue_read(q, h_err.data(), err_buf, 0, num_threads * sizeof(uint32_t), 1, &launch_ev, nullptr));
  CHECK(vx_enqueue_read(q, &h_shapes, shape_buf, 0, sizeof(uint32_t), 1, &launch_ev, nullptr));
  CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, h_dst.size() * sizeof(uint32_t), 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

  std::cout << "shapes=0x" << std::hex << h_shapes << std::dec << std::endl;
  int errors = 0;
  for (uint32_t t = 0; t < num_threads; ++t) {
    if (h_err[t] != 0) {
      if (errors < 16) {
        std::printf("*** thread %u: %u wrong words\n", t, h_err[t]);
      }
      errors += h_err[t];
    }
  }
  // each store's footprint is cleared after its check
  for (uint32_t i = 0; i < global_words; ++i) {
    if (h_dst[i] != 0) {
      if (errors < 16) {
        std::printf("*** global destination word %u = 0x%x\n", i, h_dst[i]);
      }
      ++errors;
    }
  }

  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_event_release(write_ev[1]);
  vx_event_release(write_ev[0]);
  vx_buffer_release(shape_buf);
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
