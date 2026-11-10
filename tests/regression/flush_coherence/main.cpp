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
// With -x, each launch loads src with a DXA tile copy, which reaches memory
// beside the dcache, instead of with the LSU.
// With -c, the rounds form one command batch with no host transfer between
// kernels: each round reads the buffer the previous round wrote (ping-pong),
// and the host only reads back after the last round. Kernel-boundary flushes
// alone must then make each round's writes visible to the next, including to
// a core whose cache still holds the lines from two rounds earlier.

#include <vortex2.h>
#ifdef VX_CFG_EXT_DXA_ENABLE
#include <dxa.h>
#endif
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
bool        chain       = false;
bool        use_dxa     = false;

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:r:b:k:cxh")) != -1) {
    switch (c) {
      case 'n': count       = (uint32_t)std::atoi(optarg); break;
      case 'r': rounds      = (uint32_t)std::atoi(optarg); break;
      case 'b': burst       = (uint32_t)std::atoi(optarg); break;
      case 'k': kernel_file = optarg;                      break;
      case 'c': chain       = true;                        break;
      case 'x': use_dxa     = true;                        break;
      default:
        std::printf("Usage: [-k kernel] [-n words] [-r rounds] [-b burst] [-c] [-x] [-h]\n");
        std::exit(c == 'h' ? 0 : -1);
    }
  }
}

uint32_t src_value(uint32_t i, uint32_t round) {
  uint32_t x = i * 0x9E3779B1u + round * 0x85EBCA77u + 1;
  x ^= x >> 15;
  return x * 0x2C1B3C6Du;
}

#ifdef VX_CFG_EXT_DXA_ENABLE
void program_dxa(vx_device_h dev, uint32_t slot, uint64_t addr, uint32_t tile) {
  if (vortex::dxa::program_1d(dev, slot, addr, count, tile, sizeof(uint32_t)) != 0) {
    std::fprintf(stderr, "FAIL: DXA descriptor %u\n", slot);
    std::exit(1);
  }
}
#endif

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
  std::printf("flush_coherence: n=%u rounds=%u burst=%u chain=%d\n", count, rounds, burst, chain);

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

  // DXA descriptor slot 0 covers src, 1 dst and 2 the chain's middle buffer.
  uint32_t lmem_size = 0;
  if (use_dxa) {
#ifdef VX_CFG_EXT_DXA_ENABLE
    if (count % block[0] != 0) {
      std::printf("Error: -x needs a word count that is a multiple of the block size (%u)\n", block[0]);
      return -1;
    }
    lmem_size = block[0] * sizeof(uint32_t);
    program_dxa(dev, 0, arg.src_addr, block[0]);
    program_dxa(dev, 1, arg.dst_addr, block[0]);
#else
    std::printf("Error: -x needs a build with VX_CFG_EXT_DXA_ENABLE\n");
    return -1;
#endif
  }

  // Enqueued writes and launches read their host data asynchronously, so
  // each round gets its own source buffer and argument block.
  std::vector<std::vector<uint32_t>> h_src(rounds, std::vector<uint32_t>(count));
  std::vector<kernel_arg_t> args(rounds, arg);
  std::vector<uint32_t> h_dst(count), h_acc(count, 0);
  std::vector<uint32_t> ref_dst(count), ref_acc(count, 0);
  std::vector<uint32_t> h_tail(count * burst), ref_tail(count * burst);
  CHECK(vx_enqueue_write(q, acc_buf, 0, h_acc.data(), bytes, 0, nullptr, nullptr));

  int errors = 0;
  vx_buffer_h mid_buf = nullptr;
  if (chain) {
    // round r reads the buffer round r-1 wrote: src, then dst and mid in turn
    CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &mid_buf));
    uint64_t pp_addr[2];
    CHECK(vx_buffer_address(dst_buf, &pp_addr[0]));
    CHECK(vx_buffer_address(mid_buf, &pp_addr[1]));
#ifdef VX_CFG_EXT_DXA_ENABLE
    if (use_dxa) {
      program_dxa(dev, 2, pp_addr[1], block[0]);
    }
#endif
    auto& src = h_src[0];
    for (uint32_t i = 0; i < count; ++i) {
      src[i]     = src_value(i, 0);
      ref_dst[i] = src[i];
    }
    CHECK(vx_enqueue_write(q, src_buf, 0, src.data(), bytes, 0, nullptr, nullptr));

    std::vector<vx_launch_info_t> lis(rounds);
    std::vector<vx_command_t> cmds(rounds);
    for (uint32_t r = 0; r < rounds; ++r) {
      for (uint32_t i = 0; i < count; ++i) {
        ref_acc[i] += ref_dst[i] ^ r;
        ref_dst[i]  = ref_dst[i] * 3u + r;
      }
      args[r].round    = r;
      args[r].rotate   = use_dxa ? 0 : rotation(r);
      args[r].src_addr = (r == 0) ? arg.src_addr : pp_addr[(r - 1) % 2];
      args[r].dxa_slot = !use_dxa ? 0 : ((r == 0) ? 1 : 2 + (r - 1) % 2);
      args[r].dst_addr = pp_addr[r % 2];
      vx_launch_info_t& li = lis[r];
      li = vx_launch_info_t{};
      li.struct_size  = sizeof(li);
      li.kernel       = kern;
      li.args_host    = &args[r];
      li.args_size    = sizeof(args[r]);
      li.ndim         = 1;
      li.grid_dim[0]  = grid[0];
      li.block_dim[0] = block[0];
      li.lmem_size    = lmem_size;
      cmds[r].type        = VX_COMMAND_LAUNCH;
      cmds[r].data.launch = &li;
    }
    for (uint32_t i = 0; i < count; ++i) {
      for (uint32_t k = 0; k < burst; ++k) {
        ref_tail[k * count + i] = tail_value(i, k, rounds - 1);
      }
    }
    CHECK(vx_enqueue_commands(q, cmds.data(), rounds, 0, nullptr, nullptr));

    vx_buffer_h last_buf = ((rounds - 1) % 2 == 0) ? dst_buf : mid_buf;
    CHECK(vx_enqueue_read(q, h_dst.data(), last_buf, 0, bytes, 0, nullptr, nullptr));
    if (burst) {
      CHECK(vx_enqueue_read(q, h_tail.data(), tail_buf, 0, tail_bytes, 0, nullptr, nullptr));
    }
    CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
    errors += check("dst", rounds - 1, h_dst, ref_dst);
    if (burst) {
      errors += check("tail", rounds - 1, h_tail, ref_tail);
    }
  }
  for (uint32_t r = 0; !chain && r < rounds; ++r) {
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
    args[r].rotate   = use_dxa ? 0 : rotation(r);
    args[r].dxa_slot = use_dxa ? 1 : 0;
    vx_launch_info_t li{};
    li.struct_size  = sizeof(li);
    li.kernel       = kern;
    li.args_host    = &args[r];
    li.args_size    = sizeof(args[r]);
    li.ndim         = 1;
    li.grid_dim[0]  = grid[0];
    li.block_dim[0] = block[0];
    li.lmem_size    = lmem_size;
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

  if (mid_buf) {
    vx_buffer_release(mid_buf);
  }
  if (tail_buf) {
    vx_buffer_release(tail_buf);
  }
  vx_buffer_release(acc_buf);
  vx_buffer_release(dst_buf);
  vx_buffer_release(src_buf);
  vx_kernel_release(kern);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
