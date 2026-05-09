#include <iostream>
#include <vector>
#include <unordered_set>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include "testcases.h"
#include "common.h"

TestSuite* testSuite = nullptr;
const char* kernel_file = "kernel.vxbin";
uint32_t iters = 64;
std::unordered_set<std::string> selected;
std::unordered_set<std::string> excluded;
int testid_s = 0;
int testid_e = 0;
bool stop_on_error = true;

vx_device_h device = nullptr;
vx_buffer_h shared_buffer = nullptr;
vx_buffer_h per_hart_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex AMO regression test." << std::endl;
  std::cout << "Usage: [-t<name>: select test [-x<name>: exclude]] "
            << "[-s<id>: start id] [-e<id>: end id] [-k<kernel>] "
            << "[-n<iters>: per-hart iters] [-c: continue on error] [-h]"
            << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:x:s:e:k:ch")) != -1) {
    switch (c) {
    case 'n': iters = (uint32_t)atoi(optarg); break;
    case 't': selected.insert(optarg); break;
    case 'x': excluded.insert(optarg); break;
    case 's': testid_s = atoi(optarg); break;
    case 'e': testid_e = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'c': stop_on_error = false; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (testSuite) delete testSuite;
  if (device) {
    if (shared_buffer)   vx_mem_free(shared_buffer);
    if (per_hart_buffer) vx_mem_free(per_hart_buffer);
    if (krnl_buffer)     vx_mem_free(krnl_buffer);
    if (args_buffer)     vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (iters == 0) iters = 1;

  std::cout << "iters per hart: " << iters << std::endl;
  std::cout << "kernel: " << kernel_file << std::endl;

  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_harts = (uint32_t)(num_cores * num_warps * num_threads);
  std::cout << "num_harts: " << num_harts << std::endl;

  kernel_arg.num_harts = num_harts;
  kernel_arg.iters = iters;

  // Single shared word — sized to a cache line so AMOs stay on one
  // bank and we exercise serialization rather than bank-striping.
  const size_t shared_bytes  = 64;
  const size_t per_hart_bytes = num_harts * sizeof(uint32_t);

  RT_CHECK(vx_mem_alloc(device, shared_bytes,   VX_MEM_READ_WRITE, &shared_buffer));
  RT_CHECK(vx_mem_address(shared_buffer, &kernel_arg.shared_addr));
  RT_CHECK(vx_mem_alloc(device, per_hart_bytes, VX_MEM_READ_WRITE, &per_hart_buffer));
  RT_CHECK(vx_mem_address(per_hart_buffer, &kernel_arg.per_hart_addr));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ, &args_buffer));

  std::cout << "shared_addr=0x"   << std::hex << kernel_arg.shared_addr
            << ", per_hart_addr=0x" << kernel_arg.per_hart_addr
            << std::dec << std::endl;

  std::vector<uint32_t> shared_host(shared_bytes / sizeof(uint32_t));
  std::vector<uint32_t> per_hart_host(num_harts);

  testSuite = new TestSuite(device);
  if (testid_e == 0) testid_e = (int)testSuite->size() - 1;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  int errors = 0;
  for (int t = testid_s; t <= testid_e; ++t) {
    auto test = testSuite->get_test(t);
    auto name = test->name();

    if (!selected.empty() && selected.count(name) == 0) continue;
    if (!excluded.empty() && excluded.count(name) != 0) continue;

    // atomic_critical uses non-atomic load/store inside a lock-protected
    // critical section. That pattern requires L1↔L1 cache coherence,
    // which Vortex does not implement (each core's L1 is private; AMOs
    // commit at the LLC and only invalidate the issuing core's L1).
    // The lock itself works correctly across cores, but the unprotected
    // counter read/write inside the CS observes stale L1 copies on
    // other cores. Skip on multi-core configs.
    if (name == std::string("atomic_critical") && num_cores > 1) {
      std::cout << "Test" << t << "-" << name << " SKIPPED (multi-core)" << std::endl;
      continue;
    }

    std::cout << "Test" << t << ": " << name << std::endl;

    // Initialize buffers per the test's setup.
    std::memset(shared_host.data(), 0, shared_bytes);
    std::memset(per_hart_host.data(), 0, per_hart_bytes);
    test->setup(num_harts, shared_host.data(), per_hart_host.data());

    RT_CHECK(vx_copy_to_dev(shared_buffer,   shared_host.data(),   0, shared_bytes));
    RT_CHECK(vx_copy_to_dev(per_hart_buffer, per_hart_host.data(), 0, per_hart_bytes));

    kernel_arg.testid = (uint32_t)t;
    RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));

    {
      uint32_t n = num_harts;
      uint32_t grid_dim[1], block_dim[1];
      RT_CHECK(vx_max_occupancy_grid(device, 1, &n, grid_dim, block_dim));
      RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, 0));
    }
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    RT_CHECK(vx_copy_from_dev(shared_host.data(),   shared_buffer,   0, shared_bytes));
    RT_CHECK(vx_copy_from_dev(per_hart_host.data(), per_hart_buffer, 0, per_hart_bytes));

    int err = test->verify(num_harts, iters, shared_host.data(), per_hart_host.data());
    if (err != 0) {
      std::cout << "Test" << t << "-" << name << " FAILED (" << err << " errors)" << std::endl;
      errors += err;
      if (stop_on_error) break;
    } else {
      std::cout << "Test" << t << "-" << name << " PASSED" << std::endl;
    }
  }

  cleanup();
  return errors;
}
