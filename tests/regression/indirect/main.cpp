// Device-sourced launch arguments through the command processor.
//
// One ring batch:
//   1. produce_k writes {n, 1, 1, 0} to a record in device memory.
//   2. vecadd_k launches indirectly: grid.x = ceil(record[0] / block), so the
//      grid exists only on the device.
//   3. poison_k is predicated on record[3] > 0, which is false: it and the
//      poison_k behind it (its skip range) must not run.
//   4. scale_k depends on vecadd_k's output.
// The host never reads the record; it checks the final result only.

#include <vortex2.h>
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::printf("Error: '%s' returned %s\n", #expr,                 \
                  vx_result_string(_r));                              \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

static const char* kernel_file = "kernel.vxbin";
static uint32_t size = 16384;

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n': size = std::atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    default:
      std::printf("Usage: [-k kernel] [-n size] [-h]\n");
      std::exit(c == 'h' ? 0 : 1);
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  std::printf("indirect: size=%u\n", size);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  uint64_t has_indirect = 0;
  CHECK(vx_device_query(dev, VX_CAPS_CP_INDIRECT, &has_indirect));
  if (!has_indirect) {
    std::printf("Error: device does not support indirect launch\n");
    return 1;
  }
  uint64_t nt = 0, nw = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  const uint32_t block = uint32_t(nt * nw);
  if ((block & (block - 1)) != 0) {
    std::printf("Error: block size %u is not a power of two\n", block);
    return 1;
  }
  uint8_t block_log2 = 0;
  while ((1u << block_log2) < block) {
    ++block_log2;
  }

  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, nullptr, &q));

  const uint64_t bytes = uint64_t(size) * sizeof(uint32_t);
  vx_buffer_h a_buf, b_buf, c_buf, d_buf, rec_buf;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ, &a_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ, &b_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &c_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_WRITE, &d_buf));
  CHECK(vx_buffer_create(dev, 64, VX_MEM_READ_WRITE, &rec_buf));
  uint64_t a_addr, b_addr, c_addr, d_addr, rec_addr;
  CHECK(vx_buffer_address(a_buf, &a_addr));
  CHECK(vx_buffer_address(b_buf, &b_addr));
  CHECK(vx_buffer_address(c_buf, &c_addr));
  CHECK(vx_buffer_address(d_buf, &d_addr));
  CHECK(vx_buffer_address(rec_buf, &rec_addr));

  std::vector<uint32_t> h_a(size), h_b(size), h_d(size);
  for (uint32_t i = 0; i < size; ++i) {
    h_a[i] = i * 7 + 1;
    h_b[i] = i ^ 0x55;
  }
  std::vector<uint32_t> zeros(16, 0);
  CHECK(vx_enqueue_write(q, a_buf, 0, h_a.data(), bytes, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, b_buf, 0, h_b.data(), bytes, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, rec_buf, 0, zeros.data(), 64, 0, nullptr, nullptr));

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h k_produce, k_vecadd, k_scale, k_poison;
  CHECK(vx_module_get_kernel(mod, "produce_k", &k_produce));
  CHECK(vx_module_get_kernel(mod, "vecadd_k", &k_vecadd));
  CHECK(vx_module_get_kernel(mod, "scale_k", &k_scale));
  CHECK(vx_module_get_kernel(mod, "poison_k", &k_poison));

  produce_arg_t pa = {};
  pa.dst_addr  = rec_addr;
  pa.num_words = 4;
  pa.words[0]  = size;
  pa.words[1]  = 1;
  pa.words[2]  = 1;
  pa.words[3]  = 0;
  vec_arg_t va = {a_addr, b_addr, c_addr, size, 0};
  vec_arg_t sa = {c_addr, 0, d_addr, size, 3};
  vec_arg_t xa = {0, 0, c_addr, size, 0};

  auto make_launch = [&](vx_kernel_h k, const void* args, size_t args_size,
                         uint32_t grid) {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = k;
    li.args_host    = args;
    li.args_size    = args_size;
    li.ndim         = 1;
    li.grid_dim[0]  = grid;
    li.block_dim[0] = block;
    return li;
  };
  const uint32_t grid = (size + block - 1) / block;
  vx_launch_info_t li_produce = make_launch(k_produce, &pa, sizeof(pa), 1);
  li_produce.block_dim[0] = 1;
  vx_launch_info_t li_vecadd  = make_launch(k_vecadd, &va, sizeof(va), 0);
  vx_launch_info_t li_poison  = make_launch(k_poison, &xa, sizeof(xa), grid);
  vx_launch_info_t li_poison2 = make_launch(k_poison, &xa, sizeof(xa), grid);
  vx_launch_info_t li_scale   = make_launch(k_scale, &sa, sizeof(sa), grid);

  vx_launch_indirect_t ind_vecadd = {};
  ind_vecadd.struct_size   = sizeof(ind_vecadd);
  ind_vecadd.type          = VX_STRUCT_TYPE_LAUNCH_INDIRECT;
  ind_vecadd.record_addr   = rec_addr;
  ind_vecadd.grid_word[0]  = 0;
  ind_vecadd.grid_shift[0] = block_log2;
  li_vecadd.next = &ind_vecadd;

  vx_launch_indirect_t ind_poison = {};
  ind_poison.struct_size = sizeof(ind_poison);
  ind_poison.type        = VX_STRUCT_TYPE_LAUNCH_INDIRECT;
  ind_poison.pred_addr   = rec_addr + 12;
  ind_poison.pred_ref    = 0;
  ind_poison.skip_count  = 1;
  li_poison.next = &ind_poison;

  vx_command_t cmds[5] = {};
  const vx_launch_info_t* order[5] = {&li_produce, &li_vecadd, &li_poison,
                                      &li_poison2, &li_scale};
  for (int i = 0; i < 5; ++i) {
    cmds[i].type = VX_COMMAND_LAUNCH;
    cmds[i].data.launch = order[i];
  }
  CHECK(vx_enqueue_commands(q, cmds, 5, 0, nullptr, nullptr));
  CHECK(vx_enqueue_read(q, h_d.data(), d_buf, 0, bytes, 0, nullptr, nullptr));
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));

  int errors = 0;
  for (uint32_t i = 0; i < size; ++i) {
    const uint32_t ref = (h_a[i] + h_b[i]) * 3;
    if (h_d[i] != ref) {
      if (errors < 10) {
        std::printf("*** error: [%u] expected=0x%x, actual=0x%x\n", i, ref, h_d[i]);
      }
      ++errors;
    }
  }

  vx_device_dump_perf(dev, stdout);

  vx_kernel_release(k_poison);
  vx_kernel_release(k_scale);
  vx_kernel_release(k_vecadd);
  vx_kernel_release(k_produce);
  vx_module_release(mod);
  vx_buffer_release(rec_buf);
  vx_buffer_release(d_buf);
  vx_buffer_release(c_buf);
  vx_buffer_release(b_buf);
  vx_buffer_release(a_buf);
  vx_queue_release(q);
  vx_device_release(dev);

  if (errors != 0) {
    std::printf("FAILED! (%d errors)\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
