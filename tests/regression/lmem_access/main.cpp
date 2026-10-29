// lmem_access — local-memory reads that share words must each get their own
// word, stores of several lanes to one word must all land, and every thread
// keeps its order with its own stores.

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
// The LR/SC phase retries a contended store-conditional, so its instruction
// count depends on timing; -s leaves it out where instruction counts must
// match, as in the model-parity runs.
bool        skip_lrsc   = false;

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:sh")) != -1) {
    switch (c) {
      case 'n': num_iters   = std::atoi(optarg); break;
      case 'k': kernel_file = optarg;            break;
      case 's': skip_lrsc   = true;              break;
      default:
        std::cout << "Usage: [-n iterations] [-k kernel] [-s (no LR/SC phase)] [-h]" << std::endl;
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
  uint64_t nt = 0;
  uint64_t nw = 0;
  uint64_t nc = 0;
  uint64_t lmem_size = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,      &nc));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
  const uint32_t W = static_cast<uint32_t>(nt);
  const uint32_t block = static_cast<uint32_t>(nt * nw);
  const uint32_t grid = static_cast<uint32_t>(nc * 2);
  const uint32_t num_threads = block * grid;
  // shrink the iteration count until a block's per-iteration words fit
  const uint32_t cf_words = W * W;
  const uint32_t slot_words = 4 + SUB_WORDS(W);
  while (num_iters > 1 && (RO_WORDS + slot_words * nw * num_iters + cf_words + num_iters + 2) * sizeof(uint32_t) > lmem_size) {
    num_iters /= 2;
  }
  const uint32_t slots = static_cast<uint32_t>(nw) * num_iters;
  const uint32_t words = RO_WORDS + slot_words * slots + cf_words + num_iters + 2;
  std::cout << "lmem_access: iters=" << num_iters << " block=" << block << " grid=" << grid
       << " words=" << words << std::endl;
  if (words * sizeof(uint32_t) > lmem_size) {
    std::cout << "local memory too small\nFAILED!" << std::endl;
    return 1;
  }

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  vx_buffer_h err_buf = nullptr;
  vx_buffer_h dump_buf = nullptr;
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
  kernel_arg.war_base   = RO_WORDS + slots;
  kernel_arg.amo_base   = RO_WORDS + 2 * slots;
  kernel_arg.cf_base    = RO_WORDS + 3 * slots;
  kernel_arg.sub_base   = kernel_arg.cf_base + cf_words;
  kernel_arg.same_base  = kernel_arg.sub_base + SUB_WORDS(W) * slots;
  kernel_arg.xwarp_base = kernel_arg.same_base + slots;
  kernel_arg.count_word = kernel_arg.xwarp_base + num_iters;
  kernel_arg.lrsc_word  = kernel_arg.count_word + 1;
  kernel_arg.lrsc       = skip_lrsc ? 0 : 1;
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

  std::vector<uint32_t> h_err(num_threads);
  std::vector<uint32_t> h_dump(grid * words);
  vx_event_h launch_ev = nullptr;
  vx_event_h read_ev = nullptr;
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
  // every RAW and WAR word was stored by its lane, every sub-word store word
  // holds its lanes' bytes, every same-word store word the highest lane's
  // store, every cross-warp word its warp's store, and the counters every
  // thread's increments; the read-only words kept their value
  for (uint32_t b = 0; b < grid; ++b) {
    for (uint32_t i = 0; i < words; ++i) {
      uint32_t expected = ro_word(i);
      if (i >= kernel_arg.raw_base && i < kernel_arg.amo_base) {
        expected = LMEM_MARK + i;
      } else if (i >= kernel_arg.amo_base && i < kernel_arg.cf_base && VX_CFG_EXT_A_ENABLED) {
        expected += W;
      } else if (i >= kernel_arg.sub_base && i < kernel_arg.same_base) {
        uint32_t slot = (i - kernel_arg.sub_base) / SUB_WORDS(W);
        uint32_t lane0 = ((i - kernel_arg.sub_base) % SUB_WORDS(W)) * 4;
        for (uint32_t k = 0; k < 4 && lane0 + k < W; ++k) {
          expected = (expected & ~(0xffu << (8 * k))) | ((uint32_t)sub_byte(slot, lane0 + k) << (8 * k));
        }
      } else if (i >= kernel_arg.same_base && i < kernel_arg.xwarp_base) {
        expected = same_word(i - kernel_arg.same_base, W - 1);
      } else if (i >= kernel_arg.xwarp_base && i < kernel_arg.count_word) {
        expected = xwarp_word(i - kernel_arg.xwarp_base);
      } else if (i == kernel_arg.count_word && VX_CFG_EXT_A_ENABLED) {
        expected += block * num_iters;
      } else if (i == kernel_arg.lrsc_word && VX_CFG_EXT_A_ENABLED && kernel_arg.lrsc) {
        expected += block;
      }
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
