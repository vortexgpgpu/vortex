// Kernel-boundary flush coherence under back-to-back launches.
//
// Every round the host rewrites src, then launches a kernel that reads src,
// accumulates into acc and writes dst. The element-to-thread mapping is
// rotated on three rounds out of four, so a line usually moves to another
// thread and often to another core. The post-launch cache flush must:
//   - publish each launch's writes before the host reads dst or acc;
//   - invalidate every core's (and every shared level's) copy of src before
//     the next launch, or a core re-reading a line it cached would see the
//     previous round's value;
//   - make one launch's acc writes visible to the next launch's reads on
//     other cores, also when no host transfer separates them (odd rounds do
//     not read dst back).
// With -b, each thread ends on a burst of stores to a tail buffer, so the
// last stores may still be queued in front of the shared levels when the
// flushes start, and every round is checked.

#include <vortex2.h>
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t    count       = 2048;
uint32_t    rounds      = 16;
uint32_t    burst       = 0;

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:r:b:k:h")) != -1) {
    switch (c) {
      case 'n': count       = (uint32_t)std::atoi(optarg); break;
      case 'r': rounds      = (uint32_t)std::atoi(optarg); break;
      case 'b': burst       = (uint32_t)std::atoi(optarg); break;
      case 'k': kernel_file = optarg;                      break;
      default:
        std::printf("Usage: [-k kernel] [-n words] [-r rounds] [-b burst] [-h]\n");
        std::exit(c == 'h' ? 0 : -1);
    }
  }
}

uint32_t src_value(uint32_t i, uint32_t round) {
  uint32_t x = i * 0x9E3779B1u + round * 0x85EBCA77u + 1;
  x ^= x >> 15;
  return x * 0x2C1B3C6Du;
}

uint32_t rotation(uint32_t round) {
  return (round % 4 == 0) ? 0 : (round * 0x9E3779B1u) % count;
}

int check(const char* what, uint32_t round, const std::vector<uint32_t>& got,
          const std::vector<uint32_t>& ref) {
  int errors = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] != ref[i]) {
      if (errors < 8) {
        std::printf("*** round %u %s[%zu]: expected=0x%08x actual=0x%08x\n",
                    round, what, i, ref[i], got[i]);
      }
      ++errors;
    }
  }
  return errors;
}
} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);
  const uint64_t bytes = (uint64_t)count * sizeof(uint32_t);
  const uint64_t tail_bytes = bytes * burst;
  std::printf("flush_coherence: n=%u rounds=%u burst=%u\n", count, rounds, burst);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  vx_buffer_h src_buf = nullptr, dst_buf = nullptr, acc_buf = nullptr, tail_buf = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ, &src_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_WRITE, &dst_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &acc_buf));
  if (burst) {
    CHECK(vx_buffer_create(dev, tail_bytes, VX_MEM_WRITE, &tail_buf));
  }

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t arg{};
  arg.count = count;
  CHECK(vx_buffer_address(src_buf, &arg.src_addr));
  CHECK(vx_buffer_address(dst_buf, &arg.dst_addr));
  CHECK(vx_buffer_address(acc_buf, &arg.acc_addr));
  if (burst) {
    CHECK(vx_buffer_address(tail_buf, &arg.tail_addr));
  }
  arg.burst = burst;

  uint32_t grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &count, grid, block));

  // Enqueued writes and launches read their host data asynchronously, so
  // each round gets its own source buffer and argument block.
  std::vector<std::vector<uint32_t>> h_src(rounds, std::vector<uint32_t>(count));
  std::vector<kernel_arg_t> args(rounds, arg);
  std::vector<uint32_t> h_dst(count), h_acc(count, 0);
  std::vector<uint32_t> ref_dst(count), ref_acc(count, 0);
  std::vector<uint32_t> h_tail(count * burst), ref_tail(count * burst);
  CHECK(vx_enqueue_write(q, acc_buf, 0, h_acc.data(), bytes, 0, nullptr, nullptr));

  int errors = 0;
  for (uint32_t r = 0; r < rounds; ++r) {
    auto& src = h_src[r];
    for (uint32_t i = 0; i < count; ++i) {
      src[i]     = src_value(i, r);
      ref_dst[i] = src[i] * 3u + r;
      ref_acc[i] += src[i] ^ r;
      for (uint32_t k = 0; k < burst; ++k) {
        ref_tail[k * count + i] = tail_value(i, k, r);
      }
    }
    CHECK(vx_enqueue_write(q, src_buf, 0, src.data(), bytes, 0, nullptr, nullptr));

    args[r].round  = r;
    args[r].rotate = rotation(r);
    vx_launch_info_t li{};
    li.struct_size  = sizeof(li);
    li.kernel       = kern;
    li.args_host    = &args[r];
    li.args_size    = sizeof(args[r]);
    li.ndim         = 1;
    li.grid_dim[0]  = grid[0];
    li.block_dim[0] = block[0];
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, nullptr));

    if (burst || r % 2 == 0 || r + 1 == rounds) {
      CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, bytes, 0, nullptr, nullptr));
      if (burst) {
        CHECK(vx_enqueue_read(q, h_tail.data(), tail_buf, 0, tail_bytes, 0, nullptr, nullptr));
      }
      CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
      errors += check("dst", r, h_dst, ref_dst);
      if (burst) {
        errors += check("tail", r, h_tail, ref_tail);
      }
    }
  }

  CHECK(vx_enqueue_read(q, h_acc.data(), acc_buf, 0, bytes, 0, nullptr, nullptr));
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
  errors += check("acc", rounds, h_acc, ref_acc);

  if (tail_buf) {
    vx_buffer_release(tail_buf);
  }
  vx_buffer_release(acc_buf);
  vx_buffer_release(dst_buf);
  vx_buffer_release(src_buf);
  vx_kernel_release(kern);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
