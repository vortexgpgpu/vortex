#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = (_expr);                                       \
    if (_ret == 0) break;                                     \
    std::cerr << "Error: '" << #_expr << "' returned "       \
              << _ret << std::endl;                           \
    cleanup();                                                \
    std::exit(1);                                             \
  } while (false)

namespace {

const char* kernel_file = "kernel.vxbin";
uint32_t mode = SMEM_AMO_SAME_BANK;
uint32_t iters = 16;

vx_device_h device = nullptr;
vx_queue_h queue = nullptr;
vx_module_h module = nullptr;
vx_kernel_h kernel = nullptr;
vx_buffer_h old_sums_buffer = nullptr;
vx_buffer_h final_values_buffer = nullptr;

const char* mode_name(uint32_t value) {
  switch (value) {
  case SMEM_AMO_SAME_BANK: return "same_bank_amoadd";
  case SMEM_AMO_ALL_BANKS: return "independent_bank_amoadd";
  case SMEM_PRIVATE_RMW:   return "private_nonatomic_rmw";
  case SMEM_AMO_DIRECTED:  return "directed_add_zero_sub";
  default:                 return "invalid";
  }
}

void usage() {
  std::cout << "Usage: [-m mode(0..3)] [-n iterations] [-k kernel]" << std::endl;
}

void parse_args(int argc, char** argv) {
  int opt;
  while ((opt = getopt(argc, argv, "m:n:k:h")) != -1) {
    switch (opt) {
    case 'm': mode = static_cast<uint32_t>(std::strtoul(optarg, nullptr, 0)); break;
    case 'n': iters = static_cast<uint32_t>(std::strtoul(optarg, nullptr, 0)); break;
    case 'k': kernel_file = optarg; break;
    case 'h': usage(); std::exit(0);
    default: usage(); std::exit(1);
    }
  }
  if (mode > SMEM_AMO_DIRECTED || iters == 0) {
    usage();
    std::exit(1);
  }
}

void cleanup() {
  if (device)
    vx_device_dump_perf(device, stdout);
  if (old_sums_buffer) vx_buffer_release(old_sums_buffer);
  if (final_values_buffer) vx_buffer_release(final_values_buffer);
  if (kernel) vx_kernel_release(kernel);
  if (module) vx_module_release(module);
  if (queue) vx_queue_release(queue);
  if (device) vx_device_release(device);
}

uint64_t triangular(uint64_t n) {
  return n * (n - 1) / 2;
}

} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t queue_info = {
    sizeof(queue_info), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0
  };
  RT_CHECK(vx_queue_create(device, &queue_info, &queue));

  uint64_t num_warps = 0;
  uint64_t num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  const uint32_t num_harts = static_cast<uint32_t>(num_warps * num_threads);
  const uint32_t num_banks = VX_CFG_LMEM_NUM_BANKS;
  const uint32_t final_count = std::max(num_harts, num_banks);
  const uint32_t lmem_size = (num_banks + num_harts) * sizeof(uint32_t);

  RT_CHECK(vx_check_occupancy(device, num_harts, lmem_size));

  const size_t old_count = std::max<size_t>(size_t(3) * num_harts,
                                            size_t(num_harts) * iters);
  std::vector<uint32_t> old_sums(old_count, 0);
  std::vector<uint32_t> final_values(final_count, 0);
  RT_CHECK(vx_buffer_create(device, old_sums.size() * sizeof(uint32_t),
                            VX_MEM_WRITE, &old_sums_buffer));
  RT_CHECK(vx_buffer_create(device, final_values.size() * sizeof(uint32_t),
                            VX_MEM_WRITE, &final_values_buffer));

  kernel_arg_t args = {};
  args.mode = mode;
  args.iters = iters;
  args.num_harts = num_harts;
  args.num_banks = num_banks;
  RT_CHECK(vx_buffer_address(old_sums_buffer, &args.old_sums_addr));
  RT_CHECK(vx_buffer_address(final_values_buffer, &args.final_values_addr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module));
  RT_CHECK(vx_module_get_kernel(module, "smem_amo_mlp", &kernel));

  vx_launch_info_t launch = {};
  launch.struct_size = sizeof(launch);
  launch.kernel = kernel;
  launch.args_host = &args;
  launch.args_size = sizeof(args);
  launch.ndim = 1;
  launch.grid_dim[0] = 1;
  launch.block_dim[0] = num_harts;
  launch.lmem_size = lmem_size;

  vx_event_h launch_event = nullptr;
  vx_event_h old_sums_event = nullptr;
  vx_event_h final_values_event = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &launch, 0, nullptr, &launch_event));
  RT_CHECK(vx_enqueue_read(queue, old_sums.data(), old_sums_buffer, 0,
                           old_sums.size() * sizeof(uint32_t), 1,
                           &launch_event, &old_sums_event));
  RT_CHECK(vx_enqueue_read(queue, final_values.data(), final_values_buffer, 0,
                           final_values.size() * sizeof(uint32_t), 1,
                           &launch_event, &final_values_event));
  RT_CHECK(vx_event_wait_value(final_values_event, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(final_values_event);
  vx_event_release(old_sums_event);
  vx_event_release(launch_event);

  uint32_t errors = 0;
  if (mode == SMEM_AMO_DIRECTED) {
    for (uint32_t hart = 0; hart < num_harts; ++hart) {
      const uint32_t old_zero = old_sums[hart];
      const uint32_t old_inc = old_sums[num_harts + hart];
      const uint32_t old_dec = old_sums[2 * num_harts + hart];
      if (old_zero != 7 || old_inc != 7 || old_dec != 8
       || final_values[hart] != 7) {
        std::cerr << "hart " << hart << ": old_zero=" << old_zero
                  << " old_inc=" << old_inc
                  << " old_dec=" << old_dec
                  << " final=" << final_values[hart]
                  << " expected=7,7,8,7" << std::endl;
        ++errors;
      }
    }
  } else if (mode == SMEM_PRIVATE_RMW) {
    const uint32_t expected_sum = static_cast<uint32_t>(triangular(iters));
    for (uint32_t hart = 0; hart < num_harts; ++hart) {
      if (final_values[hart] != iters || old_sums[hart] != expected_sum) {
        std::cerr << "hart " << hart << ": final=" << final_values[hart]
                  << " old_sum=" << old_sums[hart]
                  << " expected_final=" << iters
                  << " expected_old_sum=" << expected_sum << std::endl;
        ++errors;
      }
    }
  } else {
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
      // Exact-permutation oracle: across all participating harts, the
      // returned old values must be precisely {0, 1, ..., ops-1}, each seen
      // once. This catches duplicated, lost, or mis-routed responses that a
      // sum-only check can alias (e.g. {0,0,3,3} sums like {0,1,2,3}).
      uint32_t participating_harts = 0;
      for (uint32_t hart = 0; hart < num_harts; ++hart) {
        const uint32_t target_bank = mode == SMEM_AMO_SAME_BANK
                                   ? 0 : hart % num_banks;
        participating_harts += (target_bank == bank);
      }
      const uint64_t operations = uint64_t(participating_harts) * iters;
      const uint32_t expected_final = static_cast<uint32_t>(operations);
      std::vector<uint32_t> seen(operations, 0);
      uint32_t range_errors = 0, dup_errors = 0;
      for (uint32_t hart = 0; hart < num_harts; ++hart) {
        const uint32_t target_bank = mode == SMEM_AMO_SAME_BANK
                                   ? 0 : hart % num_banks;
        if (target_bank != bank)
          continue;
        for (uint32_t i = 0; i < iters; ++i) {
          const uint32_t v = old_sums[size_t(hart) * iters + i];
          if (v >= operations) { ++range_errors; continue; }
          dup_errors += (seen[v]++ != 0);
        }
      }
      uint32_t missing = 0;
      for (uint64_t v = 0; v < operations; ++v)
        missing += (seen[v] == 0);
      if (final_values[bank] != expected_final
       || range_errors || dup_errors || missing) {
        std::cerr << "bank " << bank << ": final=" << final_values[bank]
                  << " expected_final=" << expected_final
                  << " out_of_range=" << range_errors
                  << " duplicates=" << dup_errors
                  << " missing=" << missing << std::endl;
        ++errors;
      }
    }
  }

  const uint64_t atomic_ops = mode == SMEM_AMO_DIRECTED
                            ? uint64_t(num_harts) * 3
                            : mode == SMEM_PRIVATE_RMW
                            ? 0 : uint64_t(num_harts) * iters;
  const uint64_t control_ops = mode == SMEM_PRIVATE_RMW
                             ? uint64_t(num_harts) * iters : 0;
  std::cout << "SMEM_AMO_MLP mode=" << mode_name(mode)
            << " harts=" << num_harts
            << " warps=" << num_warps
            << " threads=" << num_threads
            << " banks=" << num_banks
            << " iters=" << iters
            << " atomic_ops=" << atomic_ops
            << " control_rmw_ops=" << control_ops
            << " oracle=" << (errors == 0 ? "PASS" : "FAIL")
            << std::endl;

  cleanup();
  std::cout << (errors == 0 ? "PASSED!" : "FAILED!") << std::endl;
  return errors == 0 ? 0 : 1;
}
